// The report view follows the run's tier (SPEC R25; the JSON's `mode`): a quick or fast result is
// labelled a fast check, its scores as taken on the scenarios it was picked on, and nothing in it
// speaks of a holdout or of --trust-search; checked and deep results keep the holdout labels. The
// improved prompt comes right after the result, before the buttons, and the pane asks for room.

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
// What a fast tier prints: no noise, the margin over the least gain, no search scores.
const FAST = {
  ...IMPROVED,
  verified: false,
  stop: null,
  mode: 'fast',
  score_before: 0.75,
  score_after: 1.0,
  search_score_before: null,
  search_score_after: null,
  noise: null,
  margin: 0.15,
  reason: 'fast check: not verified on held-out scenarios, no noise measured',
}
const CHECKED = {
  ...FAST,
  mode: 'checked',
  verified: true,
  search_score_before: 0.6,
  search_score_after: 0.9,
  margin: 0.05,
}
const all = (result: object) => renderText(view(result))

describe('a quick or fast result', () => {
  test('is a fast check, scored on the scenarios it was picked on', () => {
    for (const mode of ['quick', 'fast']) {
      const shown = view({ ...FAST, mode })
      expect(shown.verifiedLine, mode).toBe(`NOT verified: fast check (${mode} tier)`)
      expect(shown.scores, mode).toContain(
        'score on the scenarios it was picked on (not held out): 0.75 before, 1.00 after',
      )
      expect(shown.mode, mode).toBe(`mode: ${mode}`)
      expect(shown.useLabel, mode).toBe('Use it (not verified)')
    }
  })

  test('never speaks of a holdout or of --trust-search', () => {
    for (const mode of ['quick', 'fast']) {
      const text = all({ ...FAST, mode })
      expect(text, mode).not.toMatch(/holdout/i)
      expect(text, mode).not.toContain('--trust-search')
    }
  })

  test('shows the seconds of the run when the JSON has them', () => {
    const shown = view({ ...FAST, elapsed_s: 12.4 })
    expect(shown.verifiedLine).toBe('NOT verified: fast check (fast tier, 12 s)')
    expect(shown.mode).toBe('mode: fast (12 s)')
  })

  test('gives the margin as the CLI does, with no noise line', () => {
    expect(view(FAST).margin).toBe(
      'margin: 0.15 above the least gain of 0.10 on the scenarios it was picked on ' +
        '(no noise measured)',
    )
    expect(all(FAST)).not.toContain('noise 0.')
  })

  test('takes the meaning line from the JSON when it is there', () => {
    expect(view({ ...FAST, meaning: 'read it before you use it' }).meaning).toBe(
      'read it before you use it',
    )
    expect(view(FAST).meaning).toContain('not verified on held-out scenarios')
  })
})

describe('a checked or deep result', () => {
  test('checked: holdout score on the target model, picked-on search score', () => {
    const shown = view(CHECKED)
    expect(shown.verifiedLine).toBe('verified: yes, on held-out scenarios, on the target model')
    expect(shown.scores).toContain('holdout score (target model): 0.75 before, 1.00 after')
    expect(shown.scores).toContain(
      'score on the scenarios it was picked on (not held out): 0.60 before, 0.90 after',
    )
    expect(shown.margin).toBe(
      'margin: 0.05 above the least gain of 0.20 on the held-out scenarios (no noise measured)',
    )
    expect(shown.mode).toBe('mode: checked')
  })

  test('deep: the holdout and search labels, the noise and the bar it cleared', () => {
    const shown = view({ ...IMPROVED, mode: 'deep' })
    expect(shown.scores).toContain('holdout score (target model): 0.50 before, 0.80 after')
    expect(shown.scores).toContain('search score (search model): 0.40 before, 0.90 after')
    expect(shown.margin).toContain('noise 0.02')
    expect(shown.verifiedLine).toBe('verified: yes, on held-out scenarios, on the target model')
  })

  test('deep with --trust-search keeps its own not-verified words', () => {
    const shown = view({ ...IMPROVED, mode: 'deep', verified: false, noise: null, margin: null })
    expect(shown.verifiedLine).toContain('--trust-search')
  })

  test('a kept original has no verified line', () => {
    const kept = { ...FAST, status: 'unchanged', reason_code: 'no_reliable_improvement' }
    expect(view(kept).verifiedLine).toBeNull()
    expect(view(kept).meaning).toContain('the original is kept')
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
