// The words of a finished run by its tier (SPEC R25; the JSON's `mode`), as
// src/autoimprover/report.py says them: a quick or fast result is a fast check, scored on the
// scenarios it was picked on and never verified on held-out scenarios; a checked result is
// verified on held-out scenarios on the target model; deep (or a JSON without `mode`) is the GEPA
// search of SPEC R12 to R17, with its holdout, noise and --trust-search words. A `meaning` or
// `elapsed_s` key in the JSON wins over the lines kept here. Pure: no engine, no I/O.

const UNHELD = ['quick', 'fast']
const PIPELINE = ['quick', 'fast', 'checked']
const PICKED = 'score on the scenarios it was picked on (not held out)'

// What each reason code means after the search (report.py REASON_LINES, shortened).
const MEANINGS = new Map([
  ['improved', 'a rewrite beat the original on held-out scenarios by more than the noise'],
  ['no_reliable_improvement', 'no rewrite beat the original by more than the noise; it is kept'],
  ['already_strong', 'the original already passes nearly every check, so no search ran'],
  ['no_holdout', 'fewer than 8 scenarios leave nothing to verify a result on; give --examples'],
  ['no_candidate_beat_seed', 'no rewrite beat the original on the scenarios the search used'],
  ['unconfirmed_out_of_budget', 'calls or time ran out before a rewrite was confirmed'],
])
// The same after the quick, fast and checked stages (report.py FAST_REASON_LINES, shortened).
const FAST_MEANINGS = new Map([
  [
    'no_reliable_improvement',
    'no rewrite kept the contract and beat the original clearly enough; the original is kept',
  ],
  ['no_holdout', 'no examples were left to hold out, so the original is kept; give more examples'],
  [
    'unconfirmed_out_of_budget',
    'the clock or the calls ran out before a rewrite passed every gate; the original is kept',
  ],
])
const FAST_VERIFIED = 'a rewrite beat the original on held-out scenarios, on the target model; ' +
  'no noise was measured, so a small margin is a weak signal'
const FAST_UNVERIFIED = 'a rewrite kept the intent contract and passed the free gates in a ' +
  'short run; it is not verified on held-out scenarios and no noise was measured, so read it ' +
  'before you use it'
const DEEP_UNVERIFIED = "a rewrite scored higher on the search's own validation set; there is " +
  'no holdout and no noise was measured, so the gain is not verified'
const TRUST_SEARCH = 'NOT verified on a holdout: with --trust-search it only beat the original ' +
  'on the scenarios the search used'

/**
 * The tier-dependent fields of a finished run's view.
 * @param {Record<string, unknown>} result the CLI's JSON object
 * @returns {{ title: string, verifiedLine: string | null, mode: string | null,
 *   meaning: string | null, scores: string[], margin: string | null }}
 */
export function tierWords(result) {
  const mode = typeof result.mode === 'string' && result.mode !== '' ? result.mode : null
  const seconds = typeof result.elapsed_s === 'number' ? Math.round(result.elapsed_s) : null
  const improved = result.status === 'improved'
  const verified = result.verified === true
  const unheld = mode !== null && UNHELD.includes(mode)
  const pipeline = mode !== null && PIPELINE.includes(mode)
  const fastCheck = `fast check (${mode} tier${seconds === null ? '' : `, ${seconds} s`})`
  return {
    title: improved
      ? `improved, ${verified ? 'verified on held-out scenarios' : 'NOT verified'}` +
        (!verified && unheld ? ' (fast check)' : '')
      : 'unchanged: the original prompt is kept',
    verifiedLine: !improved
      ? null
      : verified
      ? 'verified: yes, on held-out scenarios, on the target model'
      : unheld
      ? `NOT verified: ${fastCheck}`
      : pipeline
      ? 'NOT verified'
      : `verified: no. ${TRUST_SEARCH}`,
    mode: mode === null ? null : `mode: ${mode}${seconds === null ? '' : ` (${seconds} s)`}`,
    meaning: meaningOf(result, pipeline),
    scores: scoreLines(result, unheld, pipeline),
    margin: pipeline ? fastMargin(result) : searchMargin(result),
  }
}

/**
 * @param {Record<string, unknown>} result
 * @param {boolean} pipeline
 * @returns {string | null}
 */
function meaningOf(result, pipeline) {
  if (typeof result.meaning === 'string' && result.meaning.trim() !== '') return result.meaning
  const code = typeof result.reason_code === 'string' ? result.reason_code : ''
  if (result.status === 'improved') {
    if (pipeline) return result.verified === true ? FAST_VERIFIED : FAST_UNVERIFIED
    return result.verified === true ? (MEANINGS.get(code) ?? null) : DEEP_UNVERIFIED
  }
  return (pipeline ? FAST_MEANINGS.get(code) : undefined) ?? MEANINGS.get(code) ?? null
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
 * A fast-pipeline win: its margin over the least gain it needed (report.py `_scores`).
 * @param {Record<string, unknown>} result
 * @returns {string | null}
 */
function fastMargin(result) {
  const { score_before: before, score_after: after, margin, noise } = result
  if (typeof before !== 'number' || typeof after !== 'number' || typeof margin !== 'number') {
    return null
  }
  const held = result.verified === true ? 'held-out scenarios' : 'scenarios it was picked on'
  const measured = typeof noise === 'number' ? `noise ${noise.toFixed(2)}` : 'no noise measured'
  const least = (after - before - margin).toFixed(2)
  return `margin: ${
    margin.toFixed(2)
  } above the least gain of ${least} on the ${held} (${measured})`
}

/**
 * The search's noise between the original's two holdout runs, and the bar a result cleared.
 * @param {Record<string, unknown>} result
 * @returns {string | null}
 */
function searchMargin(result) {
  if (typeof result.noise !== 'number') return null
  const noise = `noise ${result.noise.toFixed(2)} between the original's two holdout runs`
  return typeof result.margin === 'number'
    ? `margin: cleared the bar by ${result.margin.toFixed(2)} (${noise})`
    : noise
}

/** @param {unknown} value @returns {string} */
function fixed(value) {
  return typeof value === 'number' ? value.toFixed(2) : '?'
}
