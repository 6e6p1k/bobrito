<script module>
  export const title = 'GPU Memory Allocators — A Field Guide for bobr';
</script>

<script>
  let activeTab = $state('overview');

  const tabs = [
    { id: 'overview', label: 'Overview' },
    { id: 'bobr', label: 'bobr Today' },
    { id: 'survey', label: 'The Survey' },
    { id: 'compare', label: 'Compare' },
    { id: 'static', label: 'Static Planning' },
    { id: 'plan', label: 'Migration Plan' },
  ];

  const allocators = [
    {
      name: 'bobr (current)',
      backing: '544 individual cudaMallocs at startup (one per block); ~584 MB eager commit',
      split: 'None — fixed-size slabs only',
      rounding: '5 tiers: 64B → 1KB → 64KB → 1MB → 16MB; >16MB silently maps to tier 4',
      streams: 'None — free reuses block immediately; latent race if sync removed',
      growth: 'New Grid linked on tier exhaustion; still one cudaMalloc per block',
      reclaim: 'Never — cudaFree only on reset()',
      threads: 'None',
      alignment: 'Implicit via tier sizes; no explicit guarantee',
      metadata: 'Block + Grid pointers; cudaMemset on every free()',
      fit: 'Poor — ignores static graph advantage entirely',
    },
    {
      name: 'PyTorch CUDACachingAllocator',
      backing: 'Small pool (≤1MB): 2MB pages; large pool: 20MB pages; expandable_segments uses virtual reservation + on-demand physical mapping',
      split: 'Yes — maybe_split_block() on alloc; neighbors merge in expandable segments',
      rounding: 'round_size(); 1MB boundary between small/large pools',
      streams: 'Per-stream free lists; same-stream reuse is O(1) without sync; cross-stream via record_stream() + CUDA events',
      growth: 'cudaMalloc / cuMemCreate on pool miss; expandable_segments grows virtual arena',
      reclaim: 'empty_cache() returns unused segments to OS; not automatic',
      threads: 'Global mutex on pool operations',
      alignment: '512-byte minimum; tensor-core friendly',
      metadata: 'Block struct with stream, event queue, pool membership',
      fit: 'Overkill for static graph; stream model is the jewel to steal',
    },
    {
      name: 'TensorFlow BFC',
      backing: 'SubAllocator regions; allow_growth=true adds regions incrementally; false = one upfront reservation',
      split: 'Yes — split free chunk if larger than request + min split size',
      rounding: '128 logarithmic bins; min allocation 256B, multiples of 256',
      streams: 'Historically sync-oriented; XLA bypasses BFC entirely for compiled graphs',
      growth: 'New AllocationRegion from upstream on OOM; retries after coalescing cached blocks',
      reclaim: 'Region release when fully free; timing_counter for deferred coalesce',
      threads: 'Mutex on all alloc/free',
      alignment: '256-byte enforced',
      metadata: 'Chunk handles in vector; boundary-tag doubly-linked list by address',
      fit: 'Good general fallback; bobr can use simplified BFC for dynamic edge cases',
    },
    {
      name: 'NVIDIA RMM pool_memory_resource',
      backing: 'One upstream cudaMalloc for initial pool; grows to maximum_pool_size',
      split: 'Coalescing best-fit within pool',
      rounding: '256-byte alignment required',
      streams: 'Per-stream free lists; cross-stream reuse waits on recorded event',
      growth: 'Upstream allocation when pool exhausted',
      reclaim: 'Pool trim via upstream; configurable max',
      threads: 'Thread-safe by design',
      alignment: '256 bytes',
      metadata: 'Composable MR stack (pool → cuda → logging)',
      fit: 'Excellent dynamic fallback; could wrap cudaMallocAsync upstream in CUDA 12+',
    },
    {
      name: 'cudaMallocAsync / cudaMemPool',
      backing: 'Driver-managed per-device default pool; OS fetch on miss',
      split: 'Driver-internal — opaque to user',
      rounding: 'Driver-dependent; typically page-aligned',
      streams: 'Native stream ordering — alloc/free are stream commands; no manual events needed on single stream',
      growth: 'Automatic via driver; cudaMemPoolTrimTo() to reclaim',
      reclaim: 'cudaMemPoolTrimTo(minBytesToKeep); cudaDeviceGraphMemTrim for graphs',
      threads: 'Driver handles concurrency',
      alignment: 'Device default (typically 256B)',
      metadata: 'Zero user metadata — just pointer + stream discipline',
      fit: 'Best dynamic layer for bobr; replaces 90% of custom pool code',
    },
    {
      name: 'XLA / TVM Static Planner',
      backing: 'One (or few) cudaMalloc per pool at execution start; offsets computed at compile/plan time',
      split: 'N/A — logical buffers are slices in a pre-sized arena',
      rounding: 'Per-buffer alignment (often 128/256B); heap simulator packs with padding',
      streams: 'Sequential schedule makes single-stream trivial; multi-stream needs explicit lifetime edges',
      growth: 'Peak computed statically; OOM = compile-time failure (good!)',
      reclaim: 'Arena freed at session end; zero per-iteration alloc cost',
      threads: 'Plan is immutable; no alloc lock needed in hot loop',
      alignment: 'Per-color alignment in XLA BufferAssignment',
      metadata: 'Offset table per tensor; O(1) pointer = base + offset',
      fit: 'Perfect match for bobr static lazy graph — this is the primary path',
    },
  ];

  const defects = [
    { id: 1, bug: 'Silent undersize for >16MB', severity: 'critical', fix: 'Bounds check + dedicated oversize path or static plan' },
    { id: 2, bug: '~584MB eager startup reservation', severity: 'high', fix: 'Lazy grid init or single arena + static offsets' },
    { id: 3, bug: 'Internal fragmentation, no split/coalesce', severity: 'medium', fix: 'Static heap sim eliminates; or BFC/async for dynamic' },
    { id: 4, bug: 'cudaMemset on every free()', severity: 'medium', fix: 'Remove — debug builds only' },
    { id: 5, bug: 'No stream ordering', severity: 'critical', fix: 'cudaMallocAsync or event-deferred free queues' },
    { id: 6, bug: 'No thread safety', severity: 'low', fix: 'Mutex if needed; static plan avoids hot-path locks' },
    { id: 7, bug: 'reset() dangles live Block*', severity: 'high', fix: 'Generation counter or forbid reset with live tensors' },
    { id: 8, bug: 'int sizing overflow', severity: 'medium', fix: 'size_t throughout ContentData' },
    { id: 9, bug: 'One cudaMalloc per block', severity: 'high', fix: 'Single arena or driver pool' },
  ];

  const phases = [
    {
      phase: 'Day 1 — Safety',
      color: 'rose',
      items: [
        'Add bounds check: reject alloc when bytes > block->size; throw with clear message',
        'Change ContentData::_size and allocate() to size_t',
        'Remove cudaMemset from free() hot path (guard behind #ifdef BOBR_ALLOC_DEBUG)',
        'Add generation counter to MemoryPool; ContentData stores gen; reset() increments and invalidates',
      ],
    },
    {
      phase: 'Phase 1 — Stream foundation',
      color: 'amber',
      items: [
        'Thread cudaStream_t through allocate/free (or adopt cudaMallocAsync/cudaFreeAsync)',
        'Single default stream initially — correctness for free-before-kernel-done',
        'Replace per-block cudaMalloc with one cudaMalloc arena OR cudaMemPool default pool',
        'Lazy tier init: allocate Grid blocks on first use, not at startup',
      ],
    },
    {
      phase: 'Phase 2 — Static buffer plan',
      color: 'emerald',
      items: [
        'After build_backward_graph(), run liveness over fwd_order ∪ bwd_order',
        'Build interference graph: edge if live ranges overlap and sizes incompatible with in-place',
        'Greedy heap simulator (decreasing size) → offset table per Node/owned_grad',
        'Single arena cudaMalloc(peak_bytes) per training session; pointers = base + offset',
        'Mark in-place ops (add into accumulator, etc.) to allow same offset for non-overlapping lifetimes',
      ],
    },
    {
      phase: 'Phase 3 — Dynamic fallback',
      color: 'sky',
      items: [
        'cudaMallocAsync pool for: graph construction, debug tensors, Python API surprises',
        'Optional: lightweight BFC over async pool for sub-page allocations',
        'empty_arena() between training runs; keep static plan cached per graph topology hash',
      ],
    },
  ];
</script>

<article class="min-h-screen bg-gradient-to-br from-slate-950 via-slate-900 to-indigo-950 text-slate-100">
  <!-- Hero -->
  <header class="relative overflow-hidden border-b border-slate-700/50 px-6 py-16 sm:px-12">
    <div class="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-indigo-500/10 via-transparent to-transparent"></div>
    <div class="relative mx-auto max-w-5xl">
      <p class="mb-3 font-mono text-sm tracking-widest text-indigo-400 uppercase">Operation Arena · bobr-framework-cuda</p>
      <h1 class="mb-4 text-4xl font-bold tracking-tight sm:text-5xl">{title}</h1>
      <p class="max-w-3xl text-lg leading-relaxed text-slate-300">
        How modern GPU frameworks manage device memory — and why bobr's static computation graph
        is a strategic advantage that makes <em>compile-time buffer planning</em> the right primary path,
        with <em>stream-ordered dynamic pools</em> as fallback.
      </p>
    </div>
  </header>

  <!-- Tab nav -->
  <nav class="sticky top-0 z-10 border-b border-slate-700/50 bg-slate-950/80 backdrop-blur-md">
    <div class="mx-auto flex max-w-5xl gap-1 overflow-x-auto px-4 py-2">
      {#each tabs as tab}
        <button
          class="shrink-0 rounded-lg px-4 py-2 text-sm font-medium transition-colors
            {activeTab === tab.id
              ? 'bg-indigo-600 text-white'
              : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'}"
          onclick={() => (activeTab = tab.id)}
        >
          {tab.label}
        </button>
      {/each}
    </div>
  </nav>

  <div class="mx-auto max-w-5xl px-6 py-10 sm:px-12">

    <!-- OVERVIEW -->
    {#if activeTab === 'overview'}
      <section class="prose prose-invert prose-indigo max-w-none">
        <h2>Why GPU allocators exist</h2>
        <p>
          Every tensor needs a contiguous block of device memory. The naïve approach —
          <code>cudaMalloc</code> on every allocation and <code>cudaFree</code> on every deallocation —
          is catastrophically slow. A single <code>cudaMalloc</code> can take <strong>milliseconds</strong>
          because it may synchronize the entire GPU and invoke the OS driver. Training loops allocate
          and free thousands of temporaries per step; frameworks therefore <em>cache</em> freed blocks
          and reuse them, trading a bit of reserved VRAM for orders-of-magnitude faster allocation.
        </p>

        <div class="not-prose my-8 grid gap-4 sm:grid-cols-3">
          <div class="rounded-xl border border-slate-700 bg-slate-800/50 p-5">
            <div class="mb-2 text-2xl">🔄</div>
            <h3 class="font-semibold text-white">Dynamic caching</h3>
            <p class="mt-1 text-sm text-slate-400">PyTorch, TF BFC, RMM — react at runtime to unknown allocation patterns. Pay metadata + search cost per alloc.</p>
          </div>
          <div class="rounded-xl border border-slate-700 bg-slate-800/50 p-5">
            <div class="mb-2 text-2xl">📐</div>
            <h3 class="font-semibold text-white">Static planning</h3>
            <p class="mt-1 text-sm text-slate-400">XLA, TVM USMP — know the full graph upfront. Compute peak memory once; execution is pointer arithmetic.</p>
          </div>
          <div class="rounded-xl border border-indigo-500/50 bg-indigo-950/30 p-5 ring-1 ring-indigo-500/20">
            <div class="mb-2 text-2xl">🦫</div>
            <h3 class="font-semibold text-indigo-300">bobr's position</h3>
            <p class="mt-1 text-sm text-slate-300">Static lazy graph with known <code>fwd_order</code> and <code>bwd_order</code> <em>before</em> the training loop. Dynamic allocators solve ignorance bobr doesn't have.</p>
          </div>
        </div>

        <h2>The three questions every allocator answers</h2>
        <ol>
          <li><strong>Where does backing memory come from?</strong> Few large arenas vs many small cudaMallocs.</li>
          <li><strong>How are freed blocks reused?</strong> Fixed slabs, split/coalesce, or pre-computed offsets.</li>
          <li><strong>When is a freed block safe to reuse?</strong> Stream ordering — the #1 correctness concern in modern async CUDA.</li>
        </ol>

        <div class="not-prose rounded-xl border border-amber-500/30 bg-amber-950/20 p-6">
          <h3 class="text-lg font-semibold text-amber-300">Verdict preview</h3>
          <p class="mt-2 text-slate-300">
            <strong>Hybrid (d)</strong> is confirmed: static buffer plan for the training hot path,
            <code>cudaMallocAsync</code> dynamic pool for everything else. This exploits bobr's
            architectural advantage while keeping a safety net for graph construction and debugging.
          </p>
        </div>
      </section>
    {/if}

    <!-- BOBR TODAY -->
    {#if activeTab === 'bobr'}
      <section>
        <h2 class="mb-6 text-2xl font-bold">bobr's current allocator</h2>

        <div class="mb-8 rounded-xl border border-slate-700 bg-slate-800/40 p-6 font-mono text-sm">
          <pre class="overflow-x-auto text-slate-300"><span class="text-indigo-400">MemoryPool</span> (singleton)
  ├── grids[0]  64B  × 256 blocks  =   16 KB
  ├── grids[1]   1KB × 256 blocks  =  256 KB
  ├── grids[2]  64KB × 128 blocks  =    8 MB
  ├── grids[3]   1MB ×  64 blocks  =   64 MB
  └── grids[4]  16MB ×  32 blocks  =  512 MB  ← 87% of startup commit
      <span class="text-rose-400">Total ≈ 584 MB committed at construction</span></pre>
        </div>

        <p class="mb-6 text-slate-300 leading-relaxed">
          Each <code>Grid</code> pre-allocates N fixed-size <code>Block</code>s via individual
          <code>cudaMalloc</code> calls. <code>pick_grid_tier(bytes)</code> rounds up to the
          nearest tier — but anything above 16MB silently falls through to tier 4, where
          <code>ContentData</code> may store a logical size larger than the 16MB physical block.
          That's a silent out-of-bounds write waiting to happen.
        </p>

        <h3 class="mb-4 text-lg font-semibold text-rose-300">Known defects</h3>
        <div class="space-y-2">
          {#each defects as d}
            <div class="flex items-start gap-3 rounded-lg border border-slate-700/80 bg-slate-800/30 px-4 py-3">
              <span class="mt-0.5 shrink-0 rounded bg-slate-700 px-2 py-0.5 font-mono text-xs text-slate-300">#{d.id}</span>
              <div class="flex-1">
                <span class="font-medium text-white">{d.bug}</span>
                <span class="ml-2 rounded-full px-2 py-0.5 text-xs
                  {d.severity === 'critical' ? 'bg-rose-500/20 text-rose-300' : d.severity === 'high' ? 'bg-amber-500/20 text-amber-300' : 'bg-slate-600/40 text-slate-400'}">
                  {d.severity}
                </span>
                <p class="mt-1 text-sm text-slate-400">→ {d.fix}</p>
              </div>
            </div>
          {/each}
        </div>

        <div class="mt-8 rounded-xl border border-slate-600 bg-slate-800/50 p-6">
          <h3 class="mb-3 font-semibold">Hot-path anatomy (today)</h3>
          <pre class="overflow-x-auto font-mono text-xs text-slate-400"><span class="text-emerald-400">allocate(bytes)</span>  → pick_grid_tier → walk Grid linked list → pop free Block
<span class="text-rose-400">free(block)</span>       → <span class="text-rose-300">cudaMemset(0)</span> → push Block to free list → erase from tracking vector

<span class="text-amber-400">No stream parameter. No event. No generation guard.</span></pre>
        </div>
      </section>
    {/if}

    <!-- SURVEY -->
    {#if activeTab === 'survey'}
      <section class="space-y-10">
        <h2 class="text-2xl font-bold">Allocator deep dives</h2>

        <!-- PyTorch -->
        <div class="rounded-xl border border-orange-500/20 bg-slate-800/30 p-6">
          <h3 class="mb-2 flex items-center gap-2 text-xl font-semibold text-orange-300">
            <span class="text-2xl">🔥</span> PyTorch CUDACachingAllocator
          </h3>
          <p class="mb-4 text-slate-300 leading-relaxed">
            The industry reference. Two separate pools — <code>small_blocks</code> (≤1MB) and
            <code>large_blocks</code> (&gt;1MB) — prevent small tensors from fragmenting large segments.
            Small pool uses 2MB physical pages; large pool uses 20MB pages.
          </p>
          <div class="grid gap-4 sm:grid-cols-2">
            <div class="rounded-lg bg-slate-900/60 p-4">
              <h4 class="mb-2 font-medium text-white">Allocation pseudocode</h4>
              <pre class="font-mono text-xs text-slate-400">malloc(size, stream):
  size = round_size(size)
  pool = size &lt; 1MB ? small : large
  block = best_fit_same_stream(pool, size)
  if not block: block = new_segment(pool, size)
  return maybe_split(block, size)</pre>
            </div>
            <div class="rounded-lg bg-slate-900/60 p-4">
              <h4 class="mb-2 font-medium text-white">Stream ordering (the jewel)</h4>
              <pre class="font-mono text-xs text-slate-400">// Same stream: reuse immediately
// (GPU serializes kernels on stream)

// Cross-stream: record event on user stream
free(block):
  for stream in block.recorded_streams:
    record_event(stream)
  defer_until_all_events_complete(block)</pre>
            </div>
          </div>
          <p class="mt-4 text-sm text-slate-400">
            <code>expandable_segments</code> (CUDA 12.1+) reserves a large virtual address range and
            maps physical pages on demand — blocks within a segment can always coalesce because there's
            no cudaMalloc boundary between them.
          </p>
        </div>

        <!-- BFC -->
        <div class="rounded-xl border border-blue-500/20 bg-slate-800/30 p-6">
          <h3 class="mb-2 flex items-center gap-2 text-xl font-semibold text-blue-300">
            <span class="text-2xl">🧱</span> TensorFlow BFC (Best-Fit with Coalescing)
          </h3>
          <p class="mb-4 text-slate-300 leading-relaxed">
            A simplified dlmalloc for GPU arenas. Memory is organized as a sequence of
            <code>Chunk</code>s in address order (boundary-tag style). Free chunks live in
            ~128 logarithmic bins. Allocation finds the smallest bin that fits, splits if
            remainder ≥ min_split_size, and on free coalesces with adjacent free neighbors.
          </p>
          <pre class="overflow-x-auto rounded-lg bg-slate-900/60 p-4 font-mono text-xs text-slate-400">TryToCoalesce(chunk):
  if prev_chunk.is_free: merge(prev, chunk)
  if next_chunk.is_free: merge(chunk, next)
  insert_into_bin(merged_chunk)

Allocate(size):
  chunk = best_fit_bin(round_up(size, 256))
  if not chunk: grow_region(); retry
  if chunk.size > size + kMinSplit: split(chunk, size)
  return chunk.ptr</pre>
          <p class="mt-4 text-sm text-slate-400">
            Key insight for bobr: TensorFlow itself bypasses BFC when XLA compiles a graph —
            BFC is the eager-mode fallback. Same pattern applies here.
          </p>
        </div>

        <!-- RMM -->
        <div class="rounded-xl border border-green-500/20 bg-slate-800/30 p-6">
          <h3 class="mb-2 flex items-center gap-2 text-xl font-semibold text-green-300">
            <span class="text-2xl">🌊</span> NVIDIA RMM
          </h3>
          <p class="mb-4 text-slate-300 leading-relaxed">
            Composable memory resources (MRs) stacked like decorators:
            <code>logging → tracking → pool → cuda</code>. The
            <code>pool_memory_resource</code> is a coalescing best-fit sub-allocator over an
            upstream resource. Critically, it maintains <strong>per-stream free lists</strong> —
            same pattern as PyTorch — and uses events for cross-stream reuse.
          </p>
          <p class="text-sm text-slate-400">
            For bobr: don't adopt RMM wholesale (heavy dependency), but copy the
            composable architecture and per-stream free list pattern if building custom dynamic fallback.
          </p>
        </div>

        <!-- cudaMallocAsync -->
        <div class="rounded-xl border border-cyan-500/30 bg-cyan-950/10 p-6 ring-1 ring-cyan-500/10">
          <h3 class="mb-2 flex items-center gap-2 text-xl font-semibold text-cyan-300">
            <span class="text-2xl">⚡</span> cudaMallocAsync / cudaMemPool
            <span class="rounded-full bg-cyan-500/20 px-2 py-0.5 text-xs">CUDA 11.2+</span>
          </h3>
          <p class="mb-4 text-slate-300 leading-relaxed">
            The driver's answer to everything above. Allocations and frees are <strong>stream-ordered
            commands</strong> — inserted into the CUDA stream like kernel launches. No manual event
            tracking needed when all work stays on one stream. The driver maintains per-device memory
            pools, grows from OS on miss, and supports <code>cudaMemPoolTrimTo()</code> for reclamation.
          </p>
          <pre class="overflow-x-auto rounded-lg bg-slate-900/60 p-4 font-mono text-xs text-slate-400">cudaStream_t stream;
cudaMallocAsync(&amp;ptr, nbytes, stream);
kernel&lt;&lt;&lt;...&gt;&gt;&gt;(ptr, stream);
cudaFreeAsync(ptr, stream);
// ptr must not be accessed outside [malloc, free] on this stream</pre>
          <div class="mt-4 rounded-lg border border-cyan-500/20 bg-cyan-950/30 p-4">
            <p class="text-sm text-cyan-200">
              <strong>Could this replace bobr's custom code entirely?</strong> For the dynamic fallback
              path: <em>yes, almost completely</em>. For the training hot path: static planning still
              wins because it eliminates alloc calls entirely — but cudaMallocAsync is the right
              implementation for graph construction, debug buffers, and anything outside the static plan.
            </p>
          </div>
        </div>
      </section>
    {/if}

    <!-- COMPARE TABLE -->
    {#if activeTab === 'compare'}
      <section>
        <h2 class="mb-6 text-2xl font-bold">Comparison matrix</h2>
        <div class="overflow-x-auto rounded-xl border border-slate-700">
          <table class="w-full min-w-[900px] text-left text-sm">
            <thead>
              <tr class="border-b border-slate-700 bg-slate-800/80">
                <th class="px-4 py-3 font-semibold text-slate-200">Allocator</th>
                <th class="px-4 py-3 font-semibold text-slate-200">Backing store</th>
                <th class="px-4 py-3 font-semibold text-slate-200">Split / coalesce</th>
                <th class="px-4 py-3 font-semibold text-slate-200">Stream ordering</th>
                <th class="px-4 py-3 font-semibold text-slate-200">Hot-path cost</th>
                <th class="px-4 py-3 font-semibold text-slate-200">bobr fit</th>
              </tr>
            </thead>
            <tbody>
              {#each allocators as a, i}
                <tr class="border-b border-slate-700/50 {i % 2 === 0 ? 'bg-slate-900/30' : ''}">
                  <td class="px-4 py-3 font-medium text-white whitespace-nowrap">{a.name}</td>
                  <td class="px-4 py-3 text-slate-400 max-w-[200px]">{a.backing}</td>
                  <td class="px-4 py-3 text-slate-400">{a.split}</td>
                  <td class="px-4 py-3 text-slate-400 max-w-[180px]">{a.streams}</td>
                  <td class="px-4 py-3 text-slate-400">{a.metadata}</td>
                  <td class="px-4 py-3 text-slate-300 max-w-[160px]">{a.fit}</td>
                </tr>
              {/each}
            </tbody>
          </table>
        </div>

        <div class="mt-8 grid gap-4 sm:grid-cols-2">
          <div class="rounded-xl border border-slate-700 p-5">
            <h3 class="mb-2 font-semibold">Options considered</h3>
            <ul class="space-y-2 text-sm text-slate-400">
              <li><strong class="text-slate-300">(a) Slab refined</strong> — cheap but keeps fragmentation + 9 cudaMallocs-per-tier problem</li>
              <li><strong class="text-slate-300">(b) BFC arena</strong> — robust dynamic allocator; ignores static graph advantage</li>
              <li><strong class="text-slate-300">(c) Static plan only</strong> — optimal peak memory; breaks on dynamic graph edits</li>
              <li><strong class="text-emerald-300">(d) Hybrid ✓</strong> — static plan for training + async pool for everything else</li>
            </ul>
          </div>
          <div class="rounded-xl border border-emerald-500/30 bg-emerald-950/20 p-5">
            <h3 class="mb-2 font-semibold text-emerald-300">Why (d) wins for bobr</h3>
            <ul class="space-y-2 text-sm text-slate-300">
              <li>Graph is built lazily but fully known before <code>train()</code></li>
              <li><code>fwd_order</code> + <code>bwd_order</code> give a total execution schedule</li>
              <li>Peak memory computable at compile time → one cudaMalloc per session</li>
              <li>Dynamic pool still needed for graph construction &amp; API edge cases</li>
              <li>Matches XLA's own split: compiled thunks use static offsets; eager uses BFC</li>
            </ul>
          </div>
        </div>
      </section>
    {/if}

    <!-- STATIC PLANNING -->
    {#if activeTab === 'static'}
      <section class="prose prose-invert max-w-none">
        <h2>Static buffer planning — bobr's secret weapon</h2>
        <p>
          Because bobr builds a <strong>static lazy graph</strong>, the entire allocation schedule
          and tensor liveness are known before the training loop runs. This is exactly what XLA's
          <code>BufferAssignment</code> and TVM's <code>UnifiedStaticMemoryPlanner</code> exploit.
          Dynamic allocators pay runtime search cost for ignorance bobr doesn't have.
        </p>

        <h3>Algorithm (weekend-implementable)</h3>
        <div class="not-prose rounded-xl border border-slate-700 bg-slate-800/40 p-6 font-mono text-sm">
          <pre class="text-slate-300"><span class="text-indigo-400">Input:</span>  execution order [n₀, n₁, …, nₖ], size(n), in_place(n)
<span class="text-indigo-400">Output:</span> offset table { n → byte_offset }, peak_bytes

<span class="text-slate-500">// 1. Liveness intervals</span>
<span class="text-emerald-400">for</span> each node n at position i:
  birth[n] = i
  death[n] = max(i, last_use_position(n))  <span class="text-slate-500">// including backward pass uses</span>

<span class="text-slate-500">// 2. Interference</span>
<span class="text-emerald-400">for</span> each pair (a, b) where intervals overlap:
  <span class="text-emerald-400">if</span> ¬can_alias_in_place(a, b): add_edge(interference_graph, a, b)

<span class="text-slate-500">// 3. Heap simulation (XLA's default heuristic)</span>
sort buffers by decreasing size
peak = 0; free_holes = []
<span class="text-emerald-400">for</span> each buffer b in sorted order:
  hole = find_smallest_fit(free_holes, size(b))
  <span class="text-emerald-400">if</span> hole: offset[b] = hole.start; split hole
  <span class="text-emerald-400">else:</span> offset[b] = peak; peak += align(size(b), 256)
  at death[b]: return [offset[b], offset[b]+size(b)) to free_holes; coalesce adjacent

<span class="text-indigo-400">return</span> offset_table, peak</pre>
        </div>

        <h3>In-place operation hazards</h3>
        <p>
          Some ops can write into an operand's buffer when the operand dies at that point
          (destination-passing style). XLA models this with <code>HloAliasAnalysis</code> — the
          end of one buffer's lifetime may coincide with the start of another at the same offset.
          bobr must mark primitives that support in-place (e.g. <code>add</code> into accumulator)
          and relax interference checks accordingly.
        </p>

        <h3>What this buys bobr</h3>
        <ul>
          <li><strong>Zero per-iteration allocation</strong> — pointers are <code>arena_base + offset</code></li>
          <li><strong>Minimal peak memory</strong> — optimal within heuristic (greedy-by-size is ~optimal for ML graphs)</li>
          <li><strong>Compile-time OOM</strong> — if peak &gt; VRAM, fail before training starts</li>
          <li><strong>No memset, no free list, no mutex</strong> in the hot loop</li>
          <li><strong>~584MB startup → exact peak</strong> — a 2MB MLP might need &lt;1MB total</li>
        </ul>

        <h3>Integration point in bobr</h3>
        <p>
          Run the planner in <code>Graph::backward()</code> after <code>compute_bwd_order()</code>,
          when both forward and backward schedules are finalized. Store offsets on each
          <code>Node</code> / <code>owned_grad</code>. Replace <code>ContentData::allocate()</code>
          with offset assignment during a one-time <code>Graph::materialize_buffers()</code> call.
        </p>
      </section>
    {/if}

    <!-- MIGRATION PLAN -->
    {#if activeTab === 'plan'}
      <section>
        <h2 class="mb-2 text-2xl font-bold">Phased migration plan</h2>
        <p class="mb-8 text-slate-400">Each phase keeps the framework compiling. Prioritize safety before architecture.</p>

        <div class="space-y-6">
          {#each phases as p}
            <div class="rounded-xl border border-slate-700 bg-slate-800/30 overflow-hidden">
              <div class="border-b border-slate-700 px-6 py-4
                {p.color === 'rose' ? 'bg-rose-950/40' : ''}
                {p.color === 'amber' ? 'bg-amber-950/40' : ''}
                {p.color === 'emerald' ? 'bg-emerald-950/40' : ''}
                {p.color === 'sky' ? 'bg-sky-950/40' : ''}">
                <h3 class="font-semibold text-white">{p.phase}</h3>
              </div>
              <ul class="divide-y divide-slate-700/50">
                {#each p.items as item}
                  <li class="flex items-start gap-3 px-6 py-3">
                    <span class="mt-1 text-emerald-400">✓</span>
                    <span class="text-slate-300">{item}</span>
                  </li>
                {/each}
              </ul>
            </div>
          {/each}
        </div>

        <div class="mt-10 rounded-xl border border-indigo-500/30 bg-indigo-950/20 p-6">
          <h3 class="mb-4 text-lg font-semibold text-indigo-300">Target architecture</h3>
          <pre class="overflow-x-auto font-mono text-sm text-slate-300">Graph::backward()
  → compute_bwd_order()
  → <span class="text-emerald-400">BufferPlanner::plan(fwd_order, bwd_order)</span>   // static offsets
  → <span class="text-emerald-400">arena = cudaMalloc(peak_bytes)</span>               // one backing store

Graph::train() loop:
  → eval nodes using arena_base + node.offset          // zero alloc/free
  → kernels on default stream                          // stream-ordered by construction

Graph construction / debug / API:
  → <span class="text-cyan-400">cudaMallocAsync(…, stream)</span>                     // dynamic fallback
  → <span class="text-cyan-400">cudaFreeAsync(…, stream)</span></pre>
        </div>

        <div class="mt-8 rounded-xl border border-slate-600 bg-slate-800/50 p-6">
          <h3 class="mb-3 font-semibold text-white">What NOT to build</h3>
          <ul class="space-y-2 text-sm text-slate-400">
            <li>✗ Full PyTorch-style caching allocator with expandable segments — overkill for single-GPU static graph</li>
            <li>✗ Multi-GPU peer pools, pinned host memory, unified memory</li>
            <li>✗ 128-bin BFC with spatial partitioning — use cudaMallocAsync instead</li>
            <li>✗ Per-tier Grid linked lists — the entire current design is replaced, not patched indefinitely</li>
          </ul>
        </div>

        <div class="mt-8 prose prose-invert max-w-none">
          <h3>Further reading</h3>
          <ul>
            <li><a href="https://zdevito.github.io/2022/08/04/cuda-caching-allocator.html" class="text-indigo-400">Zach DeVito — PyTorch CUDA Caching Allocator guide</a></li>
            <li><a href="https://developer.nvidia.com/blog/using-cuda-stream-ordered-memory-allocator-part-1/" class="text-indigo-400">NVIDIA — Stream-Ordered Memory Allocator Part 1</a></li>
            <li><a href="https://openxla.org/xla/hlo_to_thunks" class="text-indigo-400">OpenXLA — HLO to Thunks (buffer assignment pipeline)</a></li>
            <li><a href="https://docs.rapids.ai/api/librmm/stable/classrmm_1_1mr_1_1pool__memory__resource.html" class="text-indigo-400">RMM pool_memory_resource reference</a></li>
            <li><a href="https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/stream-ordered-memory-allocation.html" class="text-indigo-400">CUDA Programming Guide — Stream-Ordered Memory</a></li>
          </ul>
        </div>
      </section>
    {/if}
  </div>

  <footer class="border-t border-slate-700/50 px-6 py-8 text-center text-sm text-slate-500">
    bobr-framework-cuda · GPU allocator research note · {new Date().getFullYear()}
  </footer>
</article>
