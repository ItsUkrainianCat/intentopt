// The words of a finished run by its tier (SPEC R25; the JSON's `mode`). The CLI's JSON carries
// its own words, `meaning`, `verified_text` and `margin_text`, and the seconds of the run's clock,
// `elapsed_s`: they are shown as given (the caller removes escape and control characters). What
// has no key of its own is labelled here as src/autoimprover/report.py labels it: each score by
// where it was taken (quick and fast on the scenarios they were picked on, checked and deep on
// held-out scenarios) and the noise between the original's two runs. Pure: no engine, no I/O.

const UNHELD = ['quick', 'fast']
const PIPELINE = ['quick', 'fast', 'checked']
const PICKED = 'score on the scenarios it was picked on (not held out)'
// The least gain a rewrite needs whatever the noise: 0.1 in the quick, fast and checked tiers
// (fast_stages.FAST_MARGIN), 0.05 after the search (runner.MIN_THRESHOLD).
const LEAST_GAIN = 0.1
const SEARCH_LEAST_GAIN = 0.05

/**
 * The tier-dependent fields of a finished run's view.
 * @param {Record<string, unknown>} result the CLI's JSON object
 * @returns {{ title: string, verifiedLine: string | null, mode: string | null,
 *   meaning: string | null, scores: string[], noise: string | null, margin: string | null }}
 */
export function tierWords(result) {
  const mode = given(result.mode)
  const improved = result.status === 'improved'
  const verified = result.verified === true
  const pipeline = mode !== null && PIPELINE.includes(mode)
  const unheld = mode !== null && UNHELD.includes(mode)
  const said = given(result.verified_text)
  const margin = given(result.margin_text)
  const seconds = typeof result.elapsed_s === 'number' ? ` (${result.elapsed_s.toFixed(0)} s)` : ''
  return {
    title: !improved
      ? 'unchanged: the original prompt is kept'
      : verified
      ? 'improved, verified on held-out scenarios'
      : `improved, NOT verified${unheld ? ' (fast check)' : ''}`,
    verifiedLine: improved && said !== null ? `verified: ${said}` : null,
    mode: mode === null ? null : `mode: ${mode}${seconds}`,
    meaning: given(result.meaning),
    scores: scoreLines(result, unheld, pipeline),
    noise: noiseLine(result, pipeline, margin),
    margin: pipeline && margin !== null ? `margin: ${margin}` : null,
  }
}

/**
 * A text the JSON gives, or null.
 * @param {unknown} value
 * @returns {string | null}
 */
function given(value) {
  return typeof value === 'string' && value.trim() !== '' ? value : null
}

/**
 * The scores, each labelled where it was taken (report.py `_scores`), and the calls used.
 * @param {Record<string, unknown>} result
 * @param {boolean} unheld
 * @param {boolean} pipeline
 * @returns {string[]}
 */
function scoreLines(result, unheld, pipeline) {
  const lines = []
  /** @param {unknown} before @param {unknown} after */
  const pair = (before, after) =>
    typeof after === 'number'
      ? `${fixed(before)} before, ${after.toFixed(2)} after`
      : `${fixed(before)} for the original`
  const { score_before, score_after, search_score_before, search_score_after } = result
  if (typeof score_before === 'number') {
    const where = unheld ? PICKED : 'holdout score (target model)'
    lines.push(`${where}: ${pair(score_before, score_after)}`)
  }
  if (typeof search_score_before === 'number') {
    const where = pipeline ? PICKED : 'search score (search model)'
    lines.push(`${where}: ${pair(search_score_before, search_score_after)}`)
  }
  if (typeof result.calls_used === 'number') lines.push(`calls used: ${result.calls_used}`)
  return lines
}

/**
 * The noise between the original's two runs (report.py `_scores`): with the gain a result had
 * to make, max(least gain, 2 x noise), when there is no margin; after the search with the
 * margin's words, which the quick, fast and checked tiers show on a line of their own.
 * @param {Record<string, unknown>} result
 * @param {boolean} pipeline
 * @param {string | null} margin the JSON's `margin_text`
 * @returns {string | null}
 */
function noiseLine(result, pipeline, margin) {
  const noise = result.noise
  if (typeof noise !== 'number') return null
  const runs = pipeline ? 'two runs on the scenarios it was picked on' : 'two holdout runs'
  const said = `noise: ${noise.toFixed(2)} between the original's ${runs}`
  if (margin === null) {
    const bar = Math.max(pipeline ? LEAST_GAIN : SEARCH_LEAST_GAIN, 2 * noise)
    return `${said}; a result had to gain more than ${bar.toFixed(2)}`
  }
  return pipeline ? said : `${said}; ${margin}`
}

/** @param {unknown} value @returns {string} */
function fixed(value) {
  return typeof value === 'number' ? value.toFixed(2) : '?'
}
