// The report view follows the run's tier (SPEC R25; the JSON's `mode`) and says what the command
// line says: the JSON's own `meaning`, `verified_text` and `margin_text` and the run's seconds
// (`elapsed_s`), cleaned of escape and control characters; the scores labelled where they were
// taken, quick and fast never as holdout scores; the noise between the original's two runs. The
// improved prompt comes right after the result, before the buttons, and the pane asks for room.
// The fixtures carry the words src/autoimprover/report.py writes for each case.

import { describe, expect, test } from 'claude-code/testing'
import { paneRows } from '../../hooks/pane.js'
import { renderText, rowsOf, viewOf } from '../../hooks/report.js'
import { IMPROVED } from './world.js'

const view = (result: object) =>
  viewOf({
    stdout: `${JSON.stringify(result)}\n`,
    stderr: '',
    code: 0,
    signal: null,
    cancelled: false,
    runDir: null,
    command: 'improve',
  })
const all = (result: object) => renderText(view(result))
const FAST_REASON = 'fast check: not verified on held-out scenarios, no noise measured'
// A fast result with its noise: the gain against the gain required (report.py `_margin_text`).
const FAST = {
  ...IMPROVED,
  verified: false,
  stop: null,
  mode: 'fast',
  elapsed_s: 12.4,
  reason: FAST_REASON,
  score_before: 0.75,
  score_after: 1.0,
  search_score_before: null,
  search_score_after: null,
  noise: 0.04,
  margin: 0.15,
  meaning: 'a rewrite kept the intent contract, passed the free gates and beat the original by ' +
    'more than the noise of two runs of the original on the few scenarios it was picked on; it ' +
    'is not verified on held-out scenarios, so read it before you use it',
  verified_text: `no. NOT VERIFIED (${FAST_REASON})`,
  margin_text: 'gain 0.25 vs required 0.10 on the scenarios it was picked on',
}
// A quick result: no noise, the margin over the least gain.
const QUICK = {
  ...FAST,
  mode: 'quick',
  elapsed_s: 9,
  noise: null,
  meaning: 'a rewrite kept the intent contract and passed the free gates in a short run; it is ' +
    'not verified on held-out scenarios and no noise was measured, so read it before you use it',
  margin_text: '0.15 above the least gain of 0.10 on the scenarios it was picked on (no noise ' +
    'measured)',
}
const CHECKED = {
  ...QUICK,
  mode: 'checked',
  elapsed_s: 95,
  verified: true,
  search_score_before: 0.6,
  search_score_after: 0.9,
  margin: 0.05,
  meaning: 'a rewrite beat the original on held-out scenarios, on the target model; no noise ' +
    'was measured, so a small margin is a weak signal',
  verified_text: 'yes, on the holdout, on the target model claude-opus-5-5',
  margin_text: '0.05 above the least gain of 0.20 on the held-out scenarios (no noise measured)',
}

describe('a quick or fast result', () => {
  test("shows the CLI's verified line, mode and seconds, meaning and margin", () => {
    const shown = view(FAST)
    expect(shown.verifiedLine).toBe(`verified: no. NOT VERIFIED (${FAST_REASON})`)
    expect(shown.mode).toBe('mode: fast (12 s)')
    expect(shown.meaning).toBe(FAST.meaning)
    expect(shown.margin).toBe(`margin: ${FAST.margin_text}`)
    expect(shown.title).toBe('improved, NOT verified (fast check)')
    expect(shown.useLabel).toBe('Use it (not verified)')
  })

  test('labels its scores as taken on the scenarios it was picked on', () => {
    for (const result of [FAST, QUICK]) {
      expect(view(result).scores, result.mode).toContain(
        'score on the scenarios it was picked on (not held out): 0.75 before, 1.00 after',
      )
    }
  })

  test("shows its noise between the original's two runs on those scenarios", () => {
    expect(view(FAST).noise).toBe(
      "noise: 0.04 between the original's two runs on the scenarios it was picked on",
    )
    const kept = { ...FAST, status: 'unchanged', margin: null, margin_text: null }
    expect(view(kept).noise).toBe(
      "noise: 0.04 between the original's two runs on the scenarios it was picked on; a result " +
        'had to gain more than 0.10',
    )
    expect(view({ ...kept, noise: 0.08 }).noise).toContain('a result had to gain more than 0.16')
  })

  test('a quick result without noise has no noise line', () => {
    expect(view(QUICK).noise).toBeNull()
    expect(view(QUICK).margin).toBe(`margin: ${QUICK.margin_text}`)
  })

  test('never speaks of a holdout or of --trust-search of its own', () => {
    for (const result of [FAST, QUICK]) {
      const text = all(result)
      expect(text, result.mode).not.toMatch(/holdout/i)
      expect(text, result.mode).not.toContain('--trust-search')
    }
  })
})

describe("the CLI's words", () => {
  test('are cleaned of escape sequences and control characters', () => {
    const shown = view({
      ...FAST,
      verified_text: 'no. \u001b[1mNOT VERIFIED\u001b[0m\u0007',
      margin_text: 'gain 0.25\u202e vs required 0.10',
      meaning: 'read\u0000 it\nfirst',
    })
    expect(shown.verifiedLine).toBe('verified: no. NOT VERIFIED')
    expect(shown.margin).toBe('margin: gain 0.25 vs required 0.10')
    expect(shown.meaning).toBe('read it first')
  })

  test('without them there is no stand-in: no meaning, verified or margin line', () => {
    const bare = { ...FAST, meaning: null, verified_text: null, margin_text: null, elapsed_s: null }
    const shown = view(bare)
    expect(shown.meaning).toBeNull()
    expect(shown.verifiedLine).toBeNull()
    expect(shown.margin).toBeNull()
    expect(shown.mode).toBe('mode: fast')
  })
})

describe('a checked or deep result', () => {
  test('checked: holdout score, picked-on search score, a margin line of its own', () => {
    const shown = view(CHECKED)
    expect(shown.verifiedLine).toBe(`verified: ${CHECKED.verified_text}`)
    expect(shown.scores).toContain('holdout score (target model): 0.75 before, 1.00 after')
    expect(shown.scores).toContain(
      'score on the scenarios it was picked on (not held out): 0.60 before, 0.90 after',
    )
    expect(shown.margin).toBe(`margin: ${CHECKED.margin_text}`)
    expect(shown.mode).toBe('mode: checked (95 s)')
    expect(shown.title).toBe('improved, verified on held-out scenarios')
  })

  test('deep: the holdout and search labels; the noise line carries the margin', () => {
    const shown = view(IMPROVED)
    expect(shown.scores).toContain('holdout score (target model): 0.50 before, 0.80 after')
    expect(shown.scores).toContain('search score (search model): 0.40 before, 0.90 after')
    expect(shown.noise).toBe(
      "noise: 0.02 between the original's two holdout runs; the result cleared the bar of 0.05 " +
        'by 0.25',
    )
    expect(shown.margin).toBeNull()
    expect(shown.mode).toBe('mode: deep (1500 s)')
  })

  test('a kept original has no verified line', () => {
    const kept = { ...QUICK, status: 'unchanged', verified_text: null, margin_text: null }
    expect(view(kept).verifiedLine).toBeNull()
    expect(view(kept).title).toBe('unchanged: the original prompt is kept')
  })
})

describe('the improved prompt in the pane', () => {
  test('comes right after the result, before the details and the buttons', () => {
    const { head, tail } = rowsOf(view(FAST), false)
    expect(head.map((row) => row.kind)).toContain('code')
    expect(head.find((row) => row.kind === 'code')?.text).toBe(IMPROVED.prompt)
    expect(tail.some((row) => row.text.startsWith('score on the scenarios'))).toBe(true)
    expect(tail.some((row) => row.kind === 'code')).toBe(false)
  })

  test('the pane asks for room for the whole prompt, wrapped', () => {
    const long = Array.from({ length: 30 }, (_, i) => `Line ${i} of a long prompt.`).join('\n')
    expect(paneRows(view({ ...FAST, prompt: long }))).toBeGreaterThan(30)
    expect(paneRows(view({ ...FAST, prompt: 'w '.repeat(400) }))).toBeGreaterThan(10)
    expect(paneRows(view({ ...FAST, prompt: 'x\n'.repeat(500) }))).toBeLessThanOrEqual(120)
  })
})
