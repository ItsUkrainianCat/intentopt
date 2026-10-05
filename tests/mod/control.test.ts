// The mod's control paths (SPEC R2, R4, R21, R22, R23), every engine call stubbed by ./world.js:
// usage, a missing uv, the dry run, one run at a time, cancel and resume, clean, and the exit
// codes of failed runs; the prompt's folder is removed on every path that made one.

import { describe, expect, test } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import {
  DRY,
  finished,
  ID,
  IMPROVED,
  PROMPT_FILE,
  RUN_DIR,
  stderr,
  stdout,
  TMP,
  world,
} from './world.js'

const boot = ($: Engine) => $.session.start({ cwd: '/w', surface: 'terminal', isInteractive: true })
const TYPED = {
  origin: { kind: 'composer' },
  presentation: { isFullscreen: false, columns: 100 },
} as const
const improve = ($: Engine, args: string) => $.command.run({ command: 'improve', args, ...TYPED })
const removed = (runs: string[][]) => runs.some((run) => run.join(' ') === `rm -rf -- ${TMP}`)
const dryRuns = (runs: string[][]) => runs.filter((run) => run.includes('--dry'))
const failure = (code: number, error: string) => [
  stderr(`run folder: ${RUN_DIR}\n`),
  stderr(`error: ${error}\nrun folder: ${RUN_DIR}\nresume with: autoimprover --resume ${ID}\n`),
  stdout(`${JSON.stringify({ status: 'error', code, error, run_dir: RUN_DIR })}\n`),
]

describe('before any child', () => {
  test(
    'an empty command, an unknown flag and --help answer with usage and start nothing',
    async ($, on) => {
      const w = world(on)
      await boot($)
      const empty = await improve($, '  ')
      expect(empty.exitCode).toBe(2)
      expect(empty.text).toStartWith('error: give a prompt')
      expect((await improve($, '--json Do it.')).text).toContain('unknown flag --json')
      expect((await improve($, '--help')).text).toContain('usage:')
      expect(w.runs).toEqual([])
      expect(w.spawns).toEqual([])
    },
  )

  test('a missing uv is one clear message, and nothing is started', async ($, on) => {
    const w = world(on, { uvMissing: true })
    await boot($)
    const answer = await improve($, 'Summarise the meeting notes.')
    expect(answer.exitCode).toBe(2)
    expect(answer.text).toContain('autoimprover needs uv')
    expect(w.runs).toEqual([['uv', '--version']])
    expect(w.spawns).toEqual([])
    expect(w.writes).toEqual([])
  })
})

describe('the plan', () => {
  test(
    '--dry shows the plan, exits 0 even when a real run would refuse, starts no run',
    async ($, on) => {
      const w = world(on, { dry: { ...DRY, refusal: 'the fixed costs exceed the budget of 30' } })
      await boot($)
      const answer = await improve($, '--dry --budget 30 Summarise the meeting notes.')
      expect(answer.exitCode).toBe(0)
      expect(answer.text).toContain(
        'a real run would refuse: the fixed costs exceed the budget of 30',
      )
      expect(dryRuns(w.runs)).toHaveLength(1)
      expect(dryRuns(w.runs)[0]?.slice(5)).toEqual([
        'autoimprover',
        '--json',
        '--budget',
        '30',
        '--file',
        PROMPT_FILE,
        '--target-model',
        'claude-opus-5-5',
        '--dry',
      ])
      expect(w.spawns).toEqual([])
      expect(removed(w.runs)).toBe(true)
    },
  )

  test('a prompt above 2,000 characters gets the plan first, then the run', async ($, on) => {
    const w = world(on, { chunks: finished(IMPROVED) })
    await boot($)
    const answer = await improve($, 'x'.repeat(2001))
    expect(answer.text).toContain('a real run would start')
    await w.results()
    expect(dryRuns(w.runs)).toHaveLength(1)
    expect(w.spawns).toHaveLength(1)
    expect(w.spawns[0]?.argv).not.toContain('--dry')
  })

  test('a prompt of exactly 2,000 characters gets no plan first', async ($, on) => {
    const w = world(on, { chunks: finished(IMPROVED) })
    await boot($)
    await improve($, 'x'.repeat(2000))
    await w.results()
    expect(dryRuns(w.runs)).toEqual([])
    expect(w.spawns).toHaveLength(1)
  })

  test('a plan that refuses stops the run with exit 2', async ($, on) => {
    const w = world(on, { dry: { ...DRY, refusal: 'the state folder is inside a git repository' } })
    await boot($)
    const answer = await improve($, 'x'.repeat(2500))
    expect(answer.exitCode).toBe(2)
    expect(answer.text).toContain(
      'a real run would refuse: the state folder is inside a git repository',
    )
    expect(w.spawns).toEqual([])
    expect(removed(w.runs)).toBe(true)
  })

  test('a --file larger than 2,000 bytes gets the plan first', async ($, on) => {
    const w = world(on, { fileBytes: 4096, chunks: finished(IMPROVED) })
    await boot($)
    await improve($, '--file /d/long.txt')
    await w.results()
    expect(dryRuns(w.runs)).toHaveLength(1)
    expect(w.stats).toEqual(['/d/long.txt'])
  })
})

describe('cancel and resume', () => {
  test(
    'cancel ends the child, keeps the run resumable and removes the prompt file',
    async ($, on) => {
      let release = () => {}
      const gate = new Promise<void>((resolve) => {
        release = resolve
      })
      const end = { code: null, signal: 'SIGTERM' }
      const w = world(on, { chunks: [stderr(`run folder: ${RUN_DIR}\n`)], gate, end })
      await boot($)
      await improve($, 'Summarise the meeting notes.')
      await w.waiting
      const busy = await improve($, 'Another prompt.')
      expect(busy.exitCode).toBe(2)
      expect(busy.text).toContain(`run ${ID} is still going; /improve cancel stops it`)
      const cancelling = improve($, 'cancel')
      release()
      const cancelled = await cancelling
      expect(cancelled.text).toBe(`cancelled; resume with /improve --resume ${ID}`)
      await w.results()
      expect(removed(w.runs)).toBe(true)
      expect(w.spawns).toHaveLength(1)
      expect(w.toasts).toContain('autoimprover: run cancelled')
      await improve($, '--resume')
      await w.results(2)
      expect(w.spawns[1]?.argv.slice(5)).toEqual(['autoimprover', '--json', '--resume', ID])
    },
  )

  test('cancel with no run going says so', async ($, on) => {
    world(on)
    await boot($)
    expect((await improve($, 'cancel')).text).toBe('no autoimprover run is going')
  })

  test('--resume with no known run is refused; with an id it continues that run', async ($, on) => {
    const w = world(on, { surfaces: [], chunks: finished(IMPROVED) })
    await boot($)
    const refused = await improve($, '--resume')
    expect(refused.exitCode).toBe(2)
    expect(refused.text).toContain('no last run is known')
    const answer = await improve($, `--resume ${ID}`)
    expect(answer.exitCode).toBe(0)
    expect(w.spawns[0]?.argv.slice(5)).toEqual(['autoimprover', '--json', '--resume', ID])
    expect(w.writes).toEqual([])
  })
})

describe('clean', () => {
  test('cleans the last run by default, then forgets it', async ($, on) => {
    const w = world(on, {}, { lastRunId: ID })
    await boot($)
    const answer = await improve($, 'clean')
    expect(answer.text).toBe(`removed the run folder of ${ID}`)
    expect(w.runs.at(-1)?.slice(5)).toEqual(['autoimprover', '--json', 'clean', ID])
    expect((await improve($, 'clean')).text).toContain('no last run is known')
    expect(w.spawns).toEqual([])
  })

  test('cleans the run given by id', async ($, on) => {
    const w = world(on)
    await boot($)
    const other = '20261003-090000-0badf00d'
    expect((await improve($, `clean ${other}`)).text).toBe(`removed the run folder of ${other}`)
    expect(w.runs.at(-1)?.at(-1)).toBe(other)
  })
})

describe('failed runs, where nothing draws a pane', () => {
  test('exit 3 for a claude that is not logged in says how to log in and resume', async ($, on) => {
    const error =
      'backend failure: claude exited with code 1; result: Not logged in \u00b7 Please run /login'
    const w = world(on, { surfaces: [], chunks: failure(3, error), end: { code: 3, signal: null } })
    await boot($)
    const answer = await improve($, 'Summarise the meeting notes.')
    expect(answer.exitCode).toBe(3)
    expect(answer.text).toContain('run `claude` once in a terminal to log in')
    expect(answer.text).toContain(`resume with: /improve --resume ${ID}`)
    expect(removed(w.runs)).toBe(true)
    await improve($, '--resume')
    expect(w.spawns[1]?.argv.at(-1)).toBe(ID)
  })

  test('exit 4 when the claude session is not locked down', async ($, on) => {
    const error = 'the session is not locked down: tools: Bash'
    world(on, { surfaces: [], chunks: failure(4, error), end: { code: 4, signal: null } })
    await boot($)
    const answer = await improve($, 'Summarise the meeting notes.')
    expect(answer.exitCode).toBe(4)
    expect(answer.text).toContain('not locked down')
  })

  test(
    'a child that cannot start is exit 1, and the prompt file is still removed',
    async ($, on) => {
      const w = world(on, { surfaces: [], spawnFails: true })
      await boot($)
      const answer = await improve($, 'Summarise the meeting notes.')
      expect(answer.exitCode).toBe(1)
      expect(answer.text).toContain('ENOENT')
      expect(removed(w.runs)).toBe(true)
      expect(w.statuses.at(-1)).toBeUndefined()
    },
  )
})

describe('failed runs with a pane', () => {
  test(
    'a child that cannot start shows the failure in the pane and removes the file',
    async ($, on) => {
      const w = world(on, { spawnFails: true })
      await boot($)
      await improve($, 'Summarise the meeting notes.')
      await w.results()
      expect(removed(w.runs)).toBe(true)
      const ui = await $.ui.mount({
        plugin: 'autoimprover',
        surface: 'terminal',
        component: 'Pane',
        requestId: 'improve',
        props: {
          title: 'improve',
          isFocused: true,
          bodyColumns: 80,
          placement: 'inline',
          scroll: { offset: 0, bodyRows: 40 },
          view: {},
        },
      })
      expect(await ui.find({ type: 'Text', text: 'ENOENT' })).toBeDefined()
      expect(await ui.find({ key: 'use' })).toBeUndefined()
      await ui.unmount()
    },
  )
})
