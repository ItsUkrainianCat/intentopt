// The world beneath the mod in its flow tests: every call the mod makes on `$` answered from
// memory (process.run, process.spawn, fs, session, store, env, ui, the prompt box), each recorded.
// A test describes the child's behaviour in a `Script` and reads what the mod did in the `World`.
// Every hook is registered once here (a second registration of an event throws), so a test
// changes behaviour through the script, which the hooks read at call time.

import { mock } from 'claude-code/testing'

/** @import { On, PaneOpenArgs, ProcessSpawnChunk } from 'claude-code' */

export const ID = '20261004-132507-1a2b3c4d'
export const RUN_DIR = `/s/autoimprover/runs/${ID}`
export const TMP = '/t/autoimprover-prompt.abc123'
export const PROMPT_FILE = `${TMP}/prompt.txt`
export const VENV = '/h/u/.cache/autoimprover/venv'
/** The props a surface hands the pane's render hook. */
export const PANE = {
  title: 'improve',
  isFocused: true,
  bodyColumns: 80,
  placement: /** @type {const} */ ('inline'),
  scroll: { offset: 0, bodyRows: 40 },
  view: {},
}

/** What the CLI prints with `--json` for an improved, verified prompt (report.py). */
export const IMPROVED = {
  status: 'improved',
  prompt: 'Summarise the meeting notes for the team in exactly five bullet points.',
  verified: true,
  stop: 'budget',
  changes: ['asks for exactly five bullets'],
  reason: 'a rewrite beat the original on the holdout by more than the noise',
  reason_code: 'improved',
  diff: 'Summarise the meeting notes for the team in {+exactly+} five bullet points.',
  contract: null,
  score_before: 0.5,
  score_after: 0.8,
  search_score_before: 0.4,
  search_score_after: 0.9,
  noise: 0.02,
  margin: 0.25,
  length_ratio: 1.1,
  calls_used: 87,
  run_dir: RUN_DIR,
  mode: 'deep',
  elapsed_s: 1500,
  meaning: 'a rewrite scored higher than the original on scenarios the search never saw, by more ' +
    'than the measured noise',
  verified_text: 'yes, on the holdout, on the target model claude-opus-5-5',
  margin_text: 'the result cleared the bar of 0.05 by 0.25',
}

/** The same run with --trust-search and no holdout, in the CLI's words (report.py). */
export const TRUSTED = {
  ...IMPROVED,
  verified: false,
  noise: null,
  margin: null,
  meaning: "a rewrite scored higher than the original on the search's own validation set; there " +
    'is no holdout and no noise was measured, so the gain is not verified',
  verified_text: 'no. NOT VERIFIED on a holdout: with --trust-search the result only beat the ' +
    'original on the scenarios the search itself used',
  margin_text: null,
}

/** What the CLI prints for `--dry --json` (report.py `plan_object`). */
export const DRY = {
  status: 'dry',
  plan: {
    models: {
      task: 'claude-haiku-4-5-20251001',
      judge: 'claude-sonnet-5-5',
      reflect: 'claude-opus-5-5',
      target: 'claude-opus-5-5',
    },
    strictness: 'conservative',
    budget: 100,
    wall_clock_s: 2700,
    allow_growth: false,
    merge: false,
    seed: 0,
  },
  scenarios: 12,
  synthesised: true,
  holdout: 4,
  valset: 3,
  dataset: 5,
  calls_before_search: 16,
  calls_after_search: 18,
  search_calls: 66,
  iteration_cost: 13,
  iterations: 5,
  iterations_best: 7,
  search_clock_s: 2025,
  final_clock_s: 675,
  refusal: /** @type {string | null} */ (null),
  keeps_original: null,
}

/** @param {string} text @returns {ProcessSpawnChunk} */
export const stdout = (text) => ({ stream: 'stdout', text })
/** @param {string} text @returns {ProcessSpawnChunk} */
export const stderr = (text) => ({ stream: 'stderr', text })

/**
 * What a finished CLI prints: its run folder on stderr, then its one object on stdout.
 * @param {object} result
 * @returns {ProcessSpawnChunk[]}
 */
export function finished(result) {
  return [stderr(`run folder: ${RUN_DIR}\n`), stdout(`${JSON.stringify(result)}\n`)]
}

/**
 * @typedef {{ surfaces?: ('terminal' | 'desktop' | 'mobile' | 'vscode')[], model?: string,
 *   uvMissing?: boolean, taken?: string[], dry?: object, fileBytes?: number,
 *   chunks?: ProcessSpawnChunk[], end?: { code: number | null, signal: string | null },
 *   spawnFails?: boolean, gate?: Promise<void>, draft?: string,
 *   boxRefuses?: 'no_composer' | 'dialog' }} Script
 * @typedef {{ runs: string[][],
 *   spawns: { argv: string[], env: Record<string, string> | undefined, cwd: string | undefined }[],
 *   writes: { path: string, text: string }[], stats: string[],
 *   registered: { name: string, immediate?: true, argumentHint?: string }[],
 *   opens: PaneOpenArgs[], statuses: (string | undefined)[], toasts: string[], logs: string[],
 *   closes: string[], copies: string[], fills: { text: string, mode: string }[], reads: number,
 *   box: ('read' | 'fill')[],
 *   results: (n?: number) => Promise<void>, waiting: Promise<void> }} World
 */

/** @param {string} [out] */
function ran(out = '') {
  const value = {
    exitCode: 0,
    stdout: out,
    stderr: '',
    isStdoutTruncated: false,
    isStderrTruncated: false,
  }
  return { value }
}

/**
 * Answers every call the mod makes; `results(n)` resolves once the pane has been opened with a
 * result `n` times, `waiting` once the child has written its chunks and waits on `script.gate`.
 * @param {On} on
 * @param {Script} [script]
 * @param {Record<string, unknown>} [store]
 * @returns {World}
 */
export function world(on, script = {}, store = {}) {
  let resultsSeen = 0
  /** @type {{ n: number, resolve: () => void }[]} */
  const waiters = []
  let markWaiting = () => {}
  /** @type {World} */
  const w = {
    runs: [],
    spawns: [],
    writes: [],
    stats: [],
    registered: [],
    opens: [],
    statuses: [],
    toasts: [],
    logs: [],
    closes: [],
    copies: [],
    fills: [],
    reads: 0,
    box: [],
    results: (n = 1) =>
      new Promise((resolve) => {
        if (resultsSeen >= n) resolve()
        else waiters.push({ n, resolve })
      }),
    waiting: new Promise((resolve) => {
      markWaiting = () => resolve()
    }),
  }
  mock.env(on, { HOME: '/h/u', TMPDIR: '/t' })
  mock.store(on, store)
  on('session.start', ($, e) => ({ cwd: e.cwd }))
  on('session.model', () => ({ value: script.model ?? 'claude-opus-5-5' }))
  on('session.cwd', () => ({ value: '/w' }))
  on('session.surfaces', () => ({ value: script.surfaces ?? ['terminal'] }))
  on('command.register', ($, e) => {
    if (script.taken?.includes(e.name)) return { deny: `/${e.name} belongs to another plugin` }
    w.registered.push(e)
    return { value: { command: e.name } }
  })
  on('process.run', ($, e) => {
    const argv = [...e.argv]
    w.runs.push(argv)
    if (argv[0] === 'uv' && argv[1] === '--version') {
      return script.uvMissing ? { deny: 'spawn uv ENOENT' } : ran('uv 0.12.19\n')
    }
    if (argv[0] === 'mktemp') return ran(`${TMP}\n`)
    if (argv[0] === 'chmod' || argv[0] === 'rm') return ran()
    if (argv.includes('--dry')) return ran(`${JSON.stringify(script.dry ?? DRY)}\n`)
    if (argv.includes('clean')) return ran('{"status":"cleaned","removed":1,"skipped":0}\n')
    return { deny: `unexpected command ${argv.join(' ')}` }
  })
  on('process.spawn', async function* ($, e) {
    w.spawns.push({ argv: [...e.argv], env: e.env, cwd: e.cwd })
    if (script.spawnFails) return { deny: 'spawn uv ENOENT' }
    for (const chunk of script.chunks ?? []) yield chunk
    if (script.gate !== undefined) {
      markWaiting()
      await script.gate
    }
    return { value: script.end ?? { code: 0, signal: null } }
  })
  on('fs.write', ($, e) => {
    w.writes.push({ path: e.path, text: e.text })
    return { value: undefined }
  })
  on('fs.stat', ($, e) => {
    w.stats.push(e.path)
    return { value: { kind: 'file', size: script.fileBytes ?? 100, mtimeMs: 0, isLink: false } }
  })
  on('ui.open', ($, e) => {
    w.opens.push(e)
    if (e.focus) {
      resultsSeen += 1
      for (const waiter of waiters.filter((x) => x.n <= resultsSeen)) waiter.resolve()
    }
    return { value: { isPlaced: true } }
  })
  on('ui.close', ($, e) => {
    w.closes.push(e.id)
    return { value: undefined }
  })
  on('ui.status', ($, e) => {
    w.statuses.push(e.text)
    return { value: undefined }
  })
  on('ui.toast', ($, e) => {
    w.toasts.push(e.text)
    return { value: undefined }
  })
  on('ui.log', ($, e) => {
    w.logs.push(e.text)
    return { value: undefined }
  })
  on('ui.invalidate', () => ({ value: undefined }))
  on('ui.copy', ($, e) => {
    w.copies.push(e.text)
    return { value: { isCopied: true } }
  })
  on('prompt.read', () => {
    w.reads += 1
    w.box.push('read')
    const text = script.draft ?? ''
    return { value: { text, cursor: text.length } }
  })
  on('prompt.fill', ($, e) => {
    w.fills.push({ text: e.text, mode: e.mode })
    w.box.push('fill')
    return script.boxRefuses ? { isFilled: false, refusal: script.boxRefuses } : { isFilled: true }
  })
  return w
}
