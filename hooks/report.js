// What the mod shows of a run (SPEC R2, R21): the one JSON object the CLI prints on stdout,
// turned into a small view (title, status line, scores, margin, length ratio, reason, what
// changed, the improved prompt or null, the run folder, the resume line, the exit code) and its
// plain-text rendering, for the pane and for surfaces that draw nothing. Pure: no engine, no I/O.
// Model-written text is data: escape sequences and control characters are removed (SPEC R19).

// The fixed reason codes of the CLI's JSON (src/autoimprover/types.py REASON_CODES).
export const REASON_CODES = [
  'improved',
  'no_reliable_improvement',
  'already_strong',
  'no_holdout',
  'no_candidate_beat_seed',
  'unconfirmed_out_of_budget',
]

// What each reason code means (src/autoimprover/report.py REASON_LINES, shortened).
const MEANINGS = new Map([
  ['improved', 'a rewrite beat the original on held-out scenarios by more than the noise'],
  ['no_reliable_improvement', 'no rewrite beat the original by more than the noise; it is kept'],
  ['already_strong', 'the original already passes nearly every check, so no search ran'],
  ['no_holdout', 'fewer than 8 scenarios leave nothing to verify a result on; give --examples'],
  ['no_candidate_beat_seed', 'no rewrite beat the original on the scenarios the search used'],
  ['unconfirmed_out_of_budget', 'calls or time ran out before a rewrite was confirmed'],
])
const UNVERIFIED =
  'NOT verified on a holdout: with --trust-search it only beat the original on the scenarios ' +
  'the search used'
const ERROR_TITLES = new Map([
  [1, 'failed: internal error'],
  [2, 'refused'],
  [3, 'failed: backend failure'],
  [4, 'failed: the claude session is not locked down'],
  [130, 'interrupted'],
])
const LOGIN = /not logged in|please run \/login/i
const LOGIN_HINT =
  'claude is not logged in here: run `claude` once in a terminal to log in, then resume'
// A terminal escape sequence (CSI, OSC, or ESC and one character); the control characters other
// than newline and tab, the C1 controls, and the bidirectional overrides (as report.py).
const ESCAPE =
  /(?:\u001b\[|\u009b)[0-?]*[ -/]*[@-~]?|\u001b\][^\u0007\u001b]*(?:\u0007|\u001b\\)?|\u001b[@-Z\\-_]?/g
const CONTROL = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u202a-\u202e\u2066-\u2069]/g
// The Code element draws at most 10,000 characters; the pane shows this many of the prompt.
const SHOWN_CHARS = 9000

/**
 * @typedef {{ state: 'improved' | 'unchanged' | 'error' | 'cancelled' | 'running' | 'plan',
 *   title: string, status: string, reason: string | null, meaning: string | null,
 *   scores: string[], margin: string | null, lengthRatio: string | null, changes: string[],
 *   improved: string | null, shown: string | null, useLabel: string | null,
 *   runDir: string | null, resume: string | null, hint: string | null, plan: string[],
 *   exitCode: number, toast: string }} View
 * @typedef {{ stdout: string, stderr: string, code: number | null, signal: string | null,
 *   cancelled: boolean, runDir: string | null, command: string }} End
 * @typedef {{ kind: 'title' | 'line' | 'dim' | 'heading' | 'hint' | 'code', text: string }} Row
 */

/**
 * `text` without terminal escape sequences and control characters; newlines and tabs stay.
 * @param {string} text
 * @returns {string}
 */
export function cleanText(text) {
  return text.replace(ESCAPE, '').replace(CONTROL, '')
}

/**
 * `cleanText` on one line: every run of whitespace becomes one space.
 * @param {string} text
 * @returns {string}
 */
export function oneLine(text) {
  return cleanText(text).split(/\s+/).filter(Boolean).join(' ')
}

/**
 * The message of a thrown value.
 * @param {unknown} error
 * @returns {string}
 */
export function messageOf(error) {
  return oneLine(error instanceof Error ? error.message : String(error))
}

/**
 * The one JSON object on the CLI's stdout (SPEC R2), or null when there is none.
 * @param {string} stdout
 * @returns {Record<string, unknown> | null}
 */
export function parseObject(stdout) {
  const text = stdout.trim()
  if (!text.startsWith('{')) return null
  try {
    const value = JSON.parse(text)
    return typeof value === 'object' && value !== null && !Array.isArray(value) ? value : null
  } catch (error) {
    if (error instanceof SyntaxError) return null
    throw error
  }
}

/**
 * How a one-shot child (a plan, a clean) ended, as `viewOf` reads it.
 * @param {{ stdout: string, stderr: string, exitCode: number }} done
 * @param {string} command
 * @returns {End}
 */
export function endOf(done, command) {
  const { stdout, stderr, exitCode } = done
  return { stdout, stderr, code: exitCode, signal: null, cancelled: false, runDir: null, command }
}

/**
 * The run id: the last part of a run folder's path.
 * @param {string | null} runDir
 * @returns {string | null}
 */
export function runIdOf(runDir) {
  const id = runDir?.replace(/\/+$/, '').split('/').pop() ?? ''
  return id === '' ? null : id
}

/**
 * The view of a finished child: its JSON object, else how it ended.
 * @param {End} end
 * @returns {View}
 */
export function viewOf(end) {
  const result = parseObject(end.stdout)
  const status = result?.status
  if (result !== null && (status === 'improved' || status === 'unchanged')) {
    return outcomeView(result, end)
  }
  if (result !== null && status === 'dry') return planView(result)
  if (result !== null && status === 'error') {
    const runDir = textOr(result.run_dir, end.runDir)
    return errorView(numberOr(result.code, 1), text(result.error), { ...end, runDir })
  }
  if (end.cancelled) return cancelledView(end)
  if (end.signal !== null) return errorView(130, `the run was stopped by ${end.signal}`, end)
  const said = lastLines(end.stderr)
  return errorView(end.code || 1, said === '' ? 'the tool printed no result' : said, end)
}

/**
 * @param {Record<string, unknown>} result
 * @param {End} end
 * @returns {View}
 */
function outcomeView(result, end) {
  const improved = result.status === 'improved'
  const verified = result.verified === true
  const code = text(result.reason_code)
  const prompt = improved ? cleanText(text(result.prompt)) : null
  const changes = Array.isArray(result.changes) ? result.changes : []
  const ratio = result.length_ratio
  return {
    ...blank(end, textOr(result.run_dir, end.runDir)),
    state: improved ? 'improved' : 'unchanged',
    title: improved
      ? `improved${verified ? ', verified on held-out scenarios' : ', NOT verified'}`
      : 'unchanged: the original prompt is kept',
    status: `result: ${improved ? 'improved' : 'unchanged'} (${code})`,
    reason: oneLine(text(result.reason)),
    meaning: improved && !verified ? UNVERIFIED : (MEANINGS.get(code) ?? null),
    scores: scoreLines(result),
    margin: marginLine(result),
    lengthRatio: improved && typeof ratio === 'number'
      ? `length: ${ratio.toFixed(2)}x the original's tokens`
      : null,
    changes: changes.map((change) => oneLine(text(change)).slice(0, 300)),
    improved: prompt,
    shown: prompt === null ? null : shownPart(prompt),
    useLabel: improved ? (verified ? 'Use it' : 'Use it (not verified)') : null,
    exitCode: 0,
    toast: improved ? 'autoimprover: a prompt to review' : 'autoimprover: the original is kept',
  }
}

/**
 * @param {Record<string, unknown>} result
 * @returns {string[]}
 */
function scoreLines(result) {
  const lines = []
  /** @param {unknown} before @param {unknown} after */
  const pair = (before, after) =>
    typeof after === 'number'
      ? `${fixed(before)} before, ${after.toFixed(2)} after`
      : `${fixed(before)} for the original`
  const { score_before, score_after, search_score_before, search_score_after } = result
  if (typeof score_before === 'number') {
    lines.push(`holdout score (target model): ${pair(score_before, score_after)}`)
  }
  if (typeof search_score_before === 'number') {
    lines.push(`search score (search model): ${pair(search_score_before, search_score_after)}`)
  }
  if (typeof result.calls_used === 'number') lines.push(`calls used: ${result.calls_used}`)
  return lines
}

/**
 * @param {Record<string, unknown>} result
 * @returns {string | null}
 */
function marginLine(result) {
  if (typeof result.noise !== 'number') return null
  const noise = `noise ${result.noise.toFixed(2)} between the original's two holdout runs`
  return typeof result.margin === 'number'
    ? `margin: cleared the bar by ${result.margin.toFixed(2)} (${noise})`
    : noise
}

/**
 * The view of `--dry` (SPEC R4): the plan, and why a real run would refuse when it would.
 * @param {Record<string, unknown>} result
 * @returns {View}
 */
export function planView(result) {
  const refusal = typeof result.refusal === 'string' ? oneLine(result.refusal) : null
  return {
    ...blank(null, null),
    state: 'plan',
    title: 'plan (dry run: no model call made, nothing written)',
    status: refusal === null ? 'a real run would start' : `a real run would refuse: ${refusal}`,
    reason: refusal,
    plan: planLines(result),
    exitCode: refusal === null ? 0 : 2,
    toast: 'autoimprover: plan shown',
  }
}

/**
 * @param {Record<string, unknown>} result
 * @returns {string[]}
 */
function planLines(result) {
  const plan = /** @type {Record<string, unknown>} */ (result.plan ?? {})
  const models = /** @type {Record<string, unknown>} */ (plan.models ?? {})
  const at = (/** @type {string} */ key) => text(result[key])
  const source = result.synthesised === true ? 'synthesised by one call' : 'from --examples'
  const lines = [
    `models: task ${text(models.task)}, judge ${text(models.judge)}, ` +
    `reflection ${text(models.reflect)}, target ${text(models.target)}`,
    `budget: ${text(plan.budget)} calls; ${at('calls_before_search')} before the search, ` +
    `${at('calls_after_search')} after it, ${at('search_calls')} for the search`,
    `scenarios: ${at('scenarios')} (${source}); holdout ${at('holdout')}, ` +
    `valset ${at('valset')}, dataset ${at('dataset')}`,
    `iterations: about ${at('iterations')} to ${at('iterations_best')} (an estimate)`,
    `clock: the search may use ${minutes(result.search_clock_s)}, ` +
    `the final steps ${minutes(result.final_clock_s)}`,
  ]
  if (typeof result.keeps_original === 'string') {
    const why = oneLine(result.keeps_original)
    lines.push(`a real run would keep the original without a model call: ${why}`)
  }
  return lines
}

/**
 * The view of a failed run: the CLI's exit code and message (SPEC R2), and what to do next.
 * @param {number} code
 * @param {string} message
 * @param {End} end
 * @returns {View}
 */
export function errorView(code, message, end) {
  const said = oneLine(message)
  const login = code === 3 && LOGIN.test(`${message}\n${end.stderr}`)
  const id = runIdOf(end.runDir)
  return {
    ...blank(end, end.runDir),
    state: 'error',
    title: ERROR_TITLES.get(code) ?? `failed with exit code ${code}`,
    status: `error: ${said}`,
    reason: said,
    hint: login ? LOGIN_HINT : null,
    resume: id !== null && code !== 2 ? `/${end.command} --resume ${id}` : null,
    exitCode: code,
    toast: `autoimprover: ${ERROR_TITLES.get(code) ?? 'failed'}`,
  }
}

/**
 * @param {End} end
 * @returns {View}
 */
function cancelledView(end) {
  const id = runIdOf(end.runDir)
  return {
    ...blank(end, end.runDir),
    state: 'cancelled',
    title: 'cancelled',
    status: cancelledText(id, end.command),
    resume: id === null ? null : `/${end.command} --resume ${id}`,
    exitCode: 130,
    toast: 'autoimprover: run cancelled',
  }
}

/**
 * What `/improve cancel` answers.
 * @param {string | null} id the run id, once the run folder exists
 * @param {string} command
 * @returns {string}
 */
export function cancelledText(id, command) {
  return id === null
    ? 'cancelled before the run folder existed; nothing to resume'
    : `cancelled; resume with /${command} --resume ${id}`
}

/**
 * The view while the child runs: its last line on stderr and the run folder once known.
 * @param {{ status: string | null, runDir: string | null, command: string }} progress
 * @returns {View}
 */
export function runningView(progress) {
  return {
    ...blank(null, progress.runDir),
    state: 'running',
    title: 'running',
    status: progress.status ?? 'starting',
    hint: `/${progress.command} cancel stops it; the run stays resumable`,
    toast: 'autoimprover: running',
  }
}

/**
 * The rows of a view, top to bottom, each with how the pane draws it: the improved prompt in
 * full, or as the pane shows it (`full` false).
 * @param {View} view
 * @param {boolean} full
 * @returns {Row[]}
 */
export function rowsOf(view, full) {
  /** @type {Row[]} */
  const rows = [
    { kind: 'title', text: `autoimprover: ${view.title}` },
    { kind: 'line', text: view.status },
  ]
  /** @param {Row['kind']} kind @param {string | null} text */
  const add = (kind, text) => {
    if (text !== null) rows.push({ kind, text })
  }
  if (view.state === 'improved' || view.state === 'unchanged') add('line', `reason: ${view.reason}`)
  add('dim', view.meaning)
  for (const line of [...view.plan, ...view.scores]) add('line', line)
  add('line', view.margin)
  add('line', view.lengthRatio)
  if (view.changes.length > 0) add('heading', 'what changed:')
  for (const change of view.changes) add('line', `  - ${change}`)
  if (view.improved !== null) add('heading', 'improved prompt:')
  add('code', full ? view.improved : view.shown)
  add('hint', view.hint)
  add('dim', view.runDir === null ? null : `run folder: ${view.runDir}`)
  add('dim', view.resume === null ? null : `resume with: ${view.resume}`)
  return rows
}

/**
 * The plain-text rendering of a view, for a surface that draws no pane (SPEC R21).
 * @param {View} view
 * @returns {string}
 */
export function renderText(view) {
  return rowsOf(view, true)
    .map((row) => row.text)
    .join('\n')
}

/**
 * What `/improve clean` answers once the CLI cleaned (its `--json` object).
 * @param {Record<string, unknown>} result
 * @param {string} id
 * @returns {string}
 */
export function cleanedText(result, id) {
  if (numberOr(result.skipped, 0) > 0) {
    return `run ${id} is still running (it holds its lock), so its folder was kept`
  }
  return numberOr(result.removed, 0) > 0
    ? `removed the run folder of ${id}`
    : `there was no run folder of ${id} to remove`
}

/**
 * The last three non-blank lines of `stderr`, cleaned, on one line.
 * @param {string} stderr
 * @returns {string}
 */
export function lastLines(stderr) {
  return stderr.split('\n').map(oneLine).filter(Boolean).slice(-3).join(' / ')
}

/**
 * The fields a view kind does not set.
 * @param {End | null} end
 * @param {string | null} runDir
 * @returns {View}
 */
function blank(end, runDir) {
  return {
    state: 'error',
    title: '',
    status: '',
    reason: null,
    meaning: null,
    scores: [],
    margin: null,
    lengthRatio: null,
    changes: [],
    improved: null,
    shown: null,
    useLabel: null,
    runDir: runDir ?? end?.runDir ?? null,
    resume: null,
    hint: null,
    plan: [],
    exitCode: 1,
    toast: '',
  }
}

/**
 * The improved prompt as the pane can draw it, cut with a note when it is too long.
 * @param {string} prompt
 * @returns {string}
 */
function shownPart(prompt) {
  const points = [...prompt]
  if (points.length <= SHOWN_CHARS) return prompt
  const more = points.length - SHOWN_CHARS
  const note = `${more} more characters; Use it and Copy take the whole prompt`
  return `${points.slice(0, SHOWN_CHARS).join('')}\n... (${note})`
}

/** @param {unknown} value @returns {string} */
function text(value) {
  if (typeof value === 'string') return value
  return value === undefined || value === null ? '' : String(value)
}

/** @param {unknown} value @param {string | null} fallback @returns {string | null} */
function textOr(value, fallback) {
  return typeof value === 'string' && value !== '' ? value : fallback
}

/** @param {unknown} value @param {number} fallback @returns {number} */
function numberOr(value, fallback) {
  return typeof value === 'number' && Number.isInteger(value) ? value : fallback
}

/** @param {unknown} value @returns {string} */
function fixed(value) {
  return typeof value === 'number' ? value.toFixed(2) : '?'
}

/** @param {unknown} seconds @returns {string} */
function minutes(seconds) {
  if (typeof seconds !== 'number') return '?'
  const whole = Math.floor(seconds / 60)
  const rest = Math.round(seconds - whole * 60)
  return rest === 0 ? `${whole} min` : `${whole} min ${rest} s`
}
