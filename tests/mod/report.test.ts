// The report view of a run (SPEC R2, R21): the CLI's JSON object, or how its child ended, as the
// pane and the plain text show it; and the child's streamed output (the run folder caught early).

import { describe, expect, test } from 'claude-code/testing'
import {
  cleanedText,
  cleanText,
  parseObject,
  planView,
  REASON_CODES,
  renderText,
  rowsOf,
  runningView,
  viewOf,
} from '../../hooks/report.js'
import { Progress } from '../../hooks/stream.js'
import { DRY, ID, IMPROVED, RUN_DIR, TRUSTED } from './world.js'

// Kept in sync by hand with src/autoimprover/types.py REASON_CODES (tests/test_mod_package.py
// checks report.js against it, this list checks the view against report.js).
const CODES = [
  'improved',
  'no_reliable_improvement',
  'already_strong',
  'no_holdout',
  'no_candidate_beat_seed',
  'unconfirmed_out_of_budget',
]

const end = (result: object | string, over: object = {}) => ({
  stdout: typeof result === 'string' ? result : `${JSON.stringify(result)}\n`,
  stderr: '',
  code: 0,
  signal: null,
  cancelled: false,
  runDir: null,
  command: 'improve',
  ...over,
})
const kept = (code: string) => ({
  ...IMPROVED,
  status: 'unchanged',
  prompt: 'the original',
  reason_code: code,
  score_after: null,
  margin: null,
  meaning: `what ${code} means`,
  verified_text: null,
  margin_text: null,
})

describe('a finished run', () => {
  test('the reason codes are those of the CLI', () => {
    expect(REASON_CODES).toEqual(CODES)
  })

  test('every reason code of a kept original reads as unchanged, with its meaning', () => {
    for (const code of CODES) {
      const view = viewOf(end(kept(code)))
      expect(view.state, code).toBe('unchanged')
      expect(view.status, code).toBe(`result: unchanged (${code})`)
      expect(view.meaning, code).toBe(`what ${code} means`)
      expect(view.verifiedLine, code).toBeNull()
      expect(view.improved, code).toBeNull()
      expect(view.useLabel, code).toBeNull()
      expect(view.exitCode, code).toBe(0)
      expect(renderText(view), code).toContain(`(${code})`)
    }
  })

  test('an improved, verified result offers Use it with the improved prompt', () => {
    const view = viewOf(end(IMPROVED))
    expect(view).toMatchObject({
      state: 'improved',
      useLabel: 'Use it',
      improved: IMPROVED.prompt,
      runDir: RUN_DIR,
      exitCode: 0,
      changes: ['asks for exactly five bullets'],
      lengthRatio: "length: 1.10x the original's tokens",
      verifiedLine: 'verified: yes, on the holdout, on the target model claude-opus-5-5',
      mode: 'mode: deep (1500 s)',
      noise: "noise: 0.02 between the original's two holdout runs; " +
        'the result cleared the bar of 0.05 by 0.25',
      margin: null,
    })
    expect(view.scores).toContain('holdout score (target model): 0.50 before, 0.80 after')
    expect(view.scores).toContain('search score (search model): 0.40 before, 0.90 after')
    expect(view.scores).toContain('calls used: 87')
    expect(renderText(view)).toContain(`improved prompt:\n${IMPROVED.prompt}`)
  })

  test('an unverified (--trust-search) result says so in its button, lines and meaning', () => {
    const view = viewOf(end(TRUSTED))
    expect(view.useLabel).toBe('Use it (not verified)')
    expect(view.title).toContain('NOT verified')
    expect(view.verifiedLine).toBe(`verified: ${TRUSTED.verified_text}`)
    expect(view.meaning).toBe(TRUSTED.meaning)
    expect(view.margin).toBeNull()
    expect(view.noise).toBeNull()
  })

  test('model-written text loses escape sequences and control characters', () => {
    const prompt = 'Do X\u001b[31m now\u001b[0m\u0007 please.\n\tKeep tabs.'
    const view = viewOf(end({ ...IMPROVED, prompt, changes: ['a\u001b]0;title\u0007b\nc'] }))
    expect(view.improved).toBe('Do X now please.\n\tKeep tabs.')
    expect(view.changes).toEqual(['ab c'])
    expect(cleanText('\u202eevil\u2066')).toBe('evil')
  })

  test('a long prompt is cut for the pane and whole for Use it and the text', () => {
    const prompt = 'y'.repeat(12000)
    const view = viewOf(end({ ...IMPROVED, prompt }))
    expect(view.improved).toBe(prompt)
    expect(view.shown?.length).toBeLessThan(10000)
    expect(view.shown).toContain('3000 more characters')
    const code = (full: boolean) => rowsOf(view, full).head.find((row) => row.kind === 'code')?.text
    expect(code(true)).toBe(prompt)
    expect(code(false)).toBe(view.shown)
  })
})

describe('a failed run', () => {
  const failed = (code: number, error: string, stderr = '') =>
    viewOf(end({ status: 'error', code, error, run_dir: RUN_DIR }, { code, stderr }))

  test('exit 3 with the not-logged-in reply says to log in, and how to resume', () => {
    const view = failed(
      3,
      'backend failure: claude exited with code 1; result: Not logged in \u00b7 Please run /login',
    )
    expect(view.exitCode).toBe(3)
    expect(view.hint).toContain('run `claude` once in a terminal to log in')
    expect(view.resume).toBe(`/improve --resume ${ID}`)
    expect(renderText(view)).toContain(`resume with: /improve --resume ${ID}`)
  })

  test('exit 3 for another reason gives no login hint', () => {
    expect(failed(3, 'backend failure: three failed calls').hint).toBeNull()
  })

  test('exit 4, 2, 1 and 130 keep their codes; a refusal has no resume line', () => {
    expect(failed(4, 'the session is not locked down: tools').title).toContain('not locked down')
    expect(failed(2, '--budget: must be a whole number')).toMatchObject({
      exitCode: 2,
      resume: null,
    })
    expect(failed(1, 'internal error: KeyError').exitCode).toBe(1)
    expect(failed(130, 'interrupted')).toMatchObject({ exitCode: 130, title: 'interrupted' })
  })

  test('a cancelled child that printed nothing is cancelled, resumable', () => {
    const view = viewOf(
      end('', { code: null, signal: 'SIGTERM', cancelled: true, runDir: RUN_DIR }),
    )
    expect(view).toMatchObject({ state: 'cancelled', exitCode: 130 })
    expect(view.status).toBe(`cancelled; resume with /improve --resume ${ID}`)
  })

  test('a child killed from outside is interrupted; one that printed no JSON is an error', () => {
    const killed = viewOf(end('', { code: null, signal: 'SIGKILL', runDir: RUN_DIR }))
    expect(killed).toMatchObject({ exitCode: 130, status: 'error: the run was stopped by SIGKILL' })
    const broken = viewOf(end('', { code: 2, stderr: 'error: failed to sync\nhint: offline\n' }))
    expect(broken).toMatchObject({
      exitCode: 2,
      status: 'error: error: failed to sync / hint: offline',
    })
    expect(viewOf(end('', { code: 0 })).status).toBe('error: the tool printed no result')
  })

  test('only one JSON object counts as a result', () => {
    expect(parseObject('')).toBeNull()
    expect(parseObject('not json')).toBeNull()
    expect(parseObject('[1]')).toBeNull()
    expect(parseObject('{"a": 1}\n{"b": 2}\n')).toBeNull()
    expect(parseObject(' {"status": "dry"}\n')).toEqual({ status: 'dry' })
  })
})

describe('the plan and the other answers', () => {
  test('a plan lists the models, budget, scenarios and clock', () => {
    const view = planView(DRY)
    expect(view).toMatchObject({ state: 'plan', exitCode: 0, status: 'a real run would start' })
    const text = renderText(view)
    for (
      const part of ['target claude-opus-5-5', 'budget: 100 calls', 'holdout 4', '33 min 45 s']
    ) {
      expect(text).toContain(part)
    }
  })

  test('a plan a real run would refuse is exit 2 with the reason', () => {
    const view = viewOf(end({ ...DRY, refusal: 'the fixed costs exceed the budget of 30' }))
    expect(view.exitCode).toBe(2)
    expect(view.status).toBe('a real run would refuse: the fixed costs exceed the budget of 30')
  })

  test('clean says what it removed', () => {
    expect(cleanedText({ status: 'cleaned', removed: 1, skipped: 0 }, ID)).toBe(
      `removed the run folder of ${ID}`,
    )
    expect(cleanedText({ status: 'cleaned', removed: 0, skipped: 0 }, ID)).toContain(
      'no run folder',
    )
    expect(cleanedText({ status: 'cleaned', removed: 0, skipped: 1 }, ID)).toContain(
      'still running',
    )
  })

  test('a running view names the cancel command', () => {
    const view = runningView({ status: 'resuming run', runDir: RUN_DIR, command: 'optimize' })
    expect(view).toMatchObject({ state: 'running', status: 'resuming run', useLabel: null })
    expect(view.hint).toContain('/optimize cancel')
  })
})

describe('the streamed output', () => {
  test('stdout is collected whole; the run folder is caught once, also when split', () => {
    const progress = new Progress()
    expect(progress.push({ stream: 'stderr', text: 'models: a\nrun fol' })).toEqual({
      status: 'models: a',
      runDir: null,
    })
    expect(progress.push({ stream: 'stderr', text: `der: ${RUN_DIR}\n` })).toEqual({
      status: `run folder: ${RUN_DIR}`,
      runDir: RUN_DIR,
    })
    expect(progress.push({ stream: 'stderr', text: 'run folder: /other\n' }).runDir).toBeNull()
    progress.push({ stream: 'stdout', text: '{"status":' })
    progress.push({ stream: 'stdout', text: '"dry"}\n' })
    expect(progress.stdout).toBe('{"status":"dry"}\n')
    expect(progress.runDir).toBe(RUN_DIR)
  })

  test('the report form of the line, and a last line without a newline', () => {
    const progress = new Progress()
    const said = `run folder: ${RUN_DIR} (remove it with: autoimprover clean ${ID})`
    expect(progress.push({ stream: 'stderr', text: said }).runDir).toBeNull()
    expect(progress.finish()).toEqual({ status: said, runDir: RUN_DIR })
  })

  test('a status line has no escape sequences and is at most 160 characters', () => {
    const progress = new Progress()
    const { status } = progress.push({
      stream: 'stderr',
      text: `\u001b[1m${'z'.repeat(300)}\u001b[0m\n`,
    })
    expect(status?.length).toBe(160)
    expect(status).not.toContain('\u001b')
  })
})
