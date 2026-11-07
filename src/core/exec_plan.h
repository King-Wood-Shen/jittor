// ***************************************************************
// Copyright (c) 2023 Jittor. All Rights Reserved.
// This file is subject to the terms and conditions defined in
// file 'LICENSE.txt', which is part of this source code package.
// ***************************************************************
#pragma once
#include "core/common.h"
#include "runtime/traversal_epoch.h"
#include "runtime/backend.h"

namespace jittor {

// Working storage of `build_exec_plan`, `count_fuse` and the hold schedule:
// nothing in it is read once the plan is built. It lives in the plan so that
// a plan reused for the next batch (see `ExecPlan::clear`) plans it without
// allocating, until a batch outgrows the largest one so far. Planning a batch
// of one operator -- every `item()`, every mask's `nonzero` -- otherwise
// allocated and freed some forty vectors.
struct PlanScratch {
    vector<Node*> bfs_q;
    vector<int> roots, next, deps;
    vector<int> in_var, producer, out_var, consumers_begin, consumers;
    vector<int> readers, read_by;
    vector<char> done, fusible;
    vector<pair<int64,int64>> ready;
    vector<int> subgraph, sharegraph, sharegraph_q, shared_id;
    // count_fuse
    vector<int> father, fuse_level, fuse_queue, unvisited_consumers, group_size;
    vector<BackendId> op_backend;
    vector<char> op_scalar;
    vector<char> forced_material, half_union, bounded;
    // schedule_hold_release
    vector<int> last_use;
};

// A lightweight trace of the producer edge as the BFS first encounters a Var.
// Compared with the later per-op output snapshot, this tells whether the edge
// was already absent during collection or disappeared while the plan was
// being assembled.
struct ProducerEdgeAudit {
    uint64 plan_generation = 0;
    uint64 event_sequence = 0;
    Var* var = nullptr;
    Op* producer = nullptr;
    int slot = -1;
    bool producer_has_var = false;
    size_t thread_id_hash = 0;
};

// A failure-path record of how an Op first entered the BFS. Producer-edge
// audits only explain Var -> producer discovery; weak-sync/fetch can also
// enqueue a consumer Op by walking a Var's reverse `_outputs` edge. Keep the
// enqueue relation and scalar identities while the edge is live, then compare
// pointers only if FusedOp later reports an empty output snapshot.
struct OpEnqueueAudit {
    uint64 plan_generation = 0;
    uint64 event_sequence = 0;
    Op* op = nullptr;
    Node* source = nullptr;
    int64 source_id = -1;
    int64 op_id = -1;
    int edge_index = -1;
    // 1: Var::_inputs -> producer Op; 2: fetched Var::_outputs -> consumer Op.
    int path = 0;
    size_t thread_id_hash = 0;
};

// What the Planner hands the Runner: one batch, already cut into fused
// segments and ordered. Everything here is *indices into `ops`*, never
// pointers into the graph beyond `ops`/`all_vars` themselves, so the plan can
// be read without walking the graph a second time.
//
// The plan is a value. Building it reads the graph and writes only the batch
// numbering (`Node::batch_index` under `stamp`); it allocates nothing on the
// device, runs no kernel, and moves no memory. Everything that does is the
// Runner's job. That split is what lets `run_sync` be read as two halves and
// is the precondition for reusing a plan across steps.
struct ExecPlan {
    // Diagnostic identity for distinguishing reused plan storage from a
    // later execution that happens to use the same addresses.
    uint64 diagnostic_generation = 0;
    size_t diagnostic_builder_thread_hash = 0;
    BackendId backend = BackendId::Cpu;
    // The batch's claim on `Node::tflag`, and the stamp every index below is
    // relative to: `Node::batch_index_at(stamp)` is only answerable while this
    // epoch is alive. Compilation is its last reader, so the Runner releases
    // it before executing -- executing can destroy nodes of this batch, and a
    // node must not be destroyed while an epoch still has it marked.
    unique_ptr<TraversalEpoch> epoch;
    int64 stamp = 0;

    // The batch, split by kind and numbered: `ops[i]->batch_index_at(stamp)`
    // is `i`, likewise for `all_vars`.
    vector<Op*> ops;
    vector<Var*> all_vars;
    // The edges the batch was collected from, recorded *then*.
    //
    // The planner used to read them live, all the way through execution:
    // `load_fused_op` walks an op's inputs to build `edges` and asks each input
    // var for its producer, and `FusedOp::update_ops` walks an op's outputs to
    // decide which of a segment's outputs have to stay in memory. `Node::free()`
    // clears exactly those lists, and it runs on other threads, so a node the
    // batch had numbered could be emptied underneath the planner -- `outputs=0`
    // and the "no in-memory output" assert, or, worse, incomplete edges and a
    // kernel reading the wrong vars. `build_exec_plan` walks these same lists
    // once; recording what it saw costs one vector per op. Collection takes
    // the graph-mutation lock only while walking these lists and acquiring
    // the batch's initial Var holds; later planner phases use those holds
    // without the lock or deferral.
    //
    // Indexed by an op's batch index, flattened: op `i`'s outputs are
    // `op_outputs[op_outputs_begin[i] .. op_outputs_begin[i+1])`. One vector
    // per op was an allocation per op per batch -- with the map below, a tenth
    // of an eager decode token's executor time went to malloc and free.
    vector<Var*> op_outputs;
    vector<int> op_outputs_begin;
    // Same, per op: its input vars and the argument position each occupies on
    // the producing side (`output_t::index`), which is what separates a real
    // data edge from a control dependency (see `reverse().index < 0`).
    vector<pair<Var*, int>> op_inputs;
    vector<int> op_inputs_begin;
    // For every var the BFS touched, by its batch index (like `all_vars`), the
    // op that produces it and the slot it occupies in that op's output list --
    // that is `Var::_inputs.front().reverse().index`, read while the edge
    // still existed. A segment asks both questions to decide whether an input
    // is produced inside it (and under which output index), instead of asking
    // the var, whose `_inputs` may have been cleared by now.
    vector<pair<Op*, int>> var_producer;
    vector<ProducerEdgeAudit> producer_edge_audit;
    vector<OpEnqueueAudit> op_enqueue_audit;
    // Backward liveness `Executor::run_sync` contributes to every var above,
    // for the batch's duration -- one per var once the hold is taken, zero
    // before that and for a plan built by anything else. Phase 7 subtracts it
    // before asking whether anyone still needs a var; see the assert there.
    int batch_hold_per_var = 0;
    // The hold itself, and when each part of it may go. `release_after[rid]`
    // lists the `all_vars` indices whose last use in this batch is the
    // segment executed at queue position `rid`; the Runner drops their hold
    // once that segment has run. Held to the end of the batch, a var's memory
    // could not be freed until every kernel of the batch had run -- for a
    // backward, the whole backward -- so the peak was the sum of every
    // intermediate rather than the most alive at once: 13.07 GB against
    // PyTorch's 9.17 GB on an 8-layer Qwen3 step, and the 28-layer one did
    // not fit a 24 GB card. After its last use nothing in the batch reads the
    // node again, which is all the hold was protecting.
    vector<VarPtr>* batch_hold = nullptr;
    vector<vector<int>> release_after;
    // ops.size(), kept because the Runner reports it after `ops` has been
    // consumed.
    int op_num = 0;
    // How many of `all_vars` were the Vars the caller actually asked for.
    // They head the BFS queue, so they are exactly `all_vars[0:start_var_num]`
    // -- `count_fuse` treats them as boundaries that must stay in memory.
    int start_var_num = 0;

    // Per var (indexed like `all_vars`), whether it may stay inside a kernel:
    //   0 can be fused   1 cannot be fused
    //   2 weak shared (may still become 1 or 3 when a shared op is cut)
    //   3 strong shared (forced to be materialised)
    // FusedOp reads this through `batch_var_fused` while generating code, so
    // it must outlive execution of the batch.
    vector<int> var_fused;

    // The fused segments, identified by their union-find root, in the order
    // they will be executed. Independent segments are ordered by `Op::order`.
    vector<int> queue;
    // Every segment's ops concatenated: [000|1111|22|3333]
    // with `range` holding the split points: ^   ^    ^  ^   ^
    // `range` is indexed by position from the *end* of `queue`: the segment
    // `queue[queue.size()-1-rid]` occupies
    // `fuse_ops[rid ? range[rid-1] : 0 .. range[rid])`.
    // An op feeding several segments is duplicated into each of them rather
    // than cut out, so `fuse_ops` can be longer than `ops`.
    vector<int> fuse_ops;
    vector<int> range;

    PlanScratch scratch;

    // Empty the plan for another batch, keeping the storage of every vector.
    // `build_exec_plan` starts from here, so a plan may be built again.
    void clear();
};

// Planner: graph -> plan. `vars` is the batch's roots, already widened by the
// weak sync; `weak_sync` additionally lets the collection walk forward into
// pending consumers of a collected node, not only backward into its inputs.
// Leaves `plan.epoch` held; the caller releases it once the batch is compiled.
void build_exec_plan(vector<Var*>& vars, bool weak_sync, ExecPlan& plan,
                     vector<VarPtr>& batch_hold,
                     vector<VarPtr>& planning_hold);
// Planner-only callers (for example plan property tests) do not carry a batch
// state beyond this call; use temporary holds for the duration of planning.
void build_exec_plan(vector<Var*>& vars, bool weak_sync, ExecPlan& plan);

// A batch that failed before running all of its ops leaves the rest pending.
// Mark every unfinished var so the next sync does not plan and execute them
// again and re-raise this batch's error from a caller that built none of
// them. Outputs that already finished keep their bytes and are untouched.
void mark_failed_batch(const ExecPlan& plan);

} // jittor
