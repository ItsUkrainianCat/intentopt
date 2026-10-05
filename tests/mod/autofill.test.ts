// The end of a run and the prompt box (SPEC R21, amended 2026-10-05): an improved prompt goes
// into an empty box by itself as an editable draft; a draft the person typed is kept, and Use it
// replaces it on request; nothing is filled for a kept original, a failure, a cancel or a plan;
// nothing is ever sent. Every engine call is stubbed by ./world.js.

import { describe, expect, test } from 'claude-code/testing'
import type { Engine } from 'claude-code/testing'
import { boxPlan, boxRefused } from '../../hooks/box.js'
import { viewOf } from '../../hooks/report.js'
import { DRY, finished, ID, IMPROVED, PANE, RUN_DIR, stderr, stdout, world } from './world.js'

const TYPED = {
  origin: { kind: 'composer' },
  presentation: { isFullscreen: false, columns: 100 },
} as const
const boot = ($: Engine) => $.session.start({ cwd: '/w', surface: 'terminal', isInteractive: true })
const improve = ($: Engine, args: string) => $.command.run({ command: 'improve', args, ...TYPED })
const mount = ($: Engine) =>
  $.ui.mount({
    plugin: 'autoimprover',
    surface: 'terminal',
    component: 'Pane',
    requestId: 'improve',
    props: PANE,
  })
const UNVERIFIED = { ...IMPROVED, verified: false, noise: null, margin: null }
const IN_THE_BOX = 'improved prompt is in the prompt box: edit it and press Enter to send'
const KEPT = 'your draft was kept; press Use it to replace it'
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

describe('the words for the prompt box', () => {
  test('an empty or blank box is filled; a draft is kept', () => {
    expect(boxPlan(view(IMPROVED), '')).toMatchObject({ fill: true })
    expect(boxPlan(view(IMPROVED), ' \n\t')?.fill).toBe(true)
    expect(boxPlan(view(IMPROVED), '')?.box).toContain(IN_THE_BOX)
    expect(boxPlan(view(IMPROVED), 'my draft')).toMatchObject({ fill: false })
    expect(boxPlan(view(IMPROVED), 'my draft')?.box).toContain(KEPT)
  })

  test('an unverified result says so in the box line and the toast', () => {
    for (const draft of ['', 'my draft']) {
      const plan = boxPlan(view(UNVERIFIED), draft)
      expect(plan?.box).toMatch(/not verified/i)
      expect(plan?.toast).toMatch(/not verified/i)
    }
    expect(boxPlan(view(IMPROVED), '')?.box).not.toMatch(/not verified/i)
  })

  test('nothing for the box when there is no improved prompt', () => {
    expect(boxPlan(view({ ...IMPROVED, status: 'unchanged' }), '')).toBeNull()
    expect(boxPlan(view(DRY), '')).toBeNull()
    expect(boxPlan(view({ status: 'error', code: 3, error: 'x', run_dir: RUN_DIR }), '')).toBeNull()
  })

  test('a box that did not take the prompt says why and what to press', () => {
    const said = boxRefused(view(IMPROVED), 'dialog')
    expect(said.fill).toBe(false)
    expect(said.box).toContain('(dialog)')
    expect(said.box).toContain('Use it or Copy')
  })
})

describe('the end of a run with a pane', () => {
  test('an empty box gets the improved prompt by itself; nothing is sent', async ($, on) => {
    const w = world(on, { chunks: finished(IMPROVED) })
    await boot($)
    await improve($, 'Summarise the meeting notes.')
    await w.results()
    expect(w.reads).toBe(1)
    expect(w.fills).toEqual([{ text: IMPROVED.prompt, mode: 'replace' }])
    expect(w.toasts.join('\n')).toContain(IN_THE_BOX)
    const ui = await mount($)
    expect(await ui.find({ type: 'Text', text: IN_THE_BOX })).toBeDefined()
    expect((await ui.find({ key: 'use' }))?.text).toBe('Use it')
    expect((await ui.find({ key: 'copy' }))?.text).toBe('Copy')
    expect((await ui.find({ key: 'close' }))?.text).toBe('Close')
    expect(await ui.find({ key: 'keep' })).toBeUndefined()
    await ui.press({ key: 'use' })
    expect(w.fills).toEqual([
      { text: IMPROVED.prompt, mode: 'replace' },
      { text: IMPROVED.prompt, mode: 'replace' },
    ])
    await ui.unmount()
  })

  test('a draft the person typed is kept; Use it replaces it on request', async ($, on) => {
    const w = world(on, { chunks: finished(IMPROVED), draft: 'my own half-written prompt' })
    await boot($)
    await improve($, 'Summarise the meeting notes.')
    await w.results()
    expect(w.fills).toEqual([])
    expect(w.toasts.join('\n')).toContain(KEPT)
    const ui = await mount($)
    expect(await ui.find({ type: 'Text', text: KEPT })).toBeDefined()
    await ui.press({ key: 'use' })
    expect(w.fills).toEqual([{ text: IMPROVED.prompt, mode: 'replace' }])
    await ui.unmount()
  })

  test('an unverified result is filled too, and says not verified', async ($, on) => {
    const w = world(on, { chunks: finished(UNVERIFIED) })
    await boot($)
    await improve($, '--trust-search Summarise the meeting notes.')
    await w.results()
    expect(w.fills).toHaveLength(1)
    expect(w.toasts.join('\n')).toMatch(/not verified/i)
    const ui = await mount($)
    expect(
      await ui.find({ type: 'Text', text: /prompt box.*not verified|not verified.*prompt box/i }),
    )
      .toBeDefined()
    await ui.unmount()
  })

  test('a box that does not take it leaves Use it and Copy', async ($, on) => {
    const w = world(on, { chunks: finished(IMPROVED), boxRefuses: 'dialog' })
    await boot($)
    await improve($, 'Summarise the meeting notes.')
    await w.results()
    expect(w.fills).toHaveLength(1)
    const ui = await mount($)
    expect(await ui.find({ type: 'Text', text: 'did not take' })).toBeDefined()
    expect(await ui.find({ key: 'use' })).toBeDefined()
    await ui.unmount()
  })

  test('the Close button closes the pane and fills nothing more', async ($, on) => {
    const w = world(on, { chunks: finished(IMPROVED) })
    await boot($)
    await improve($, 'Summarise the meeting notes.')
    await w.results()
    const ui = await mount($)
    await ui.press({ key: 'close' })
    expect(w.closes).toEqual(['improve'])
    expect(w.fills).toHaveLength(1)
    await ui.unmount()
  })
})

describe('nothing is filled', () => {
  test('for a kept original', async ($, on) => {
    const kept = { ...IMPROVED, status: 'unchanged', reason_code: 'no_reliable_improvement' }
    const w = world(on, { chunks: finished(kept) })
    await boot($)
    await improve($, 'Summarise the meeting notes.')
    await w.results()
    expect(w.fills).toEqual([])
    expect(w.reads).toBe(0)
  })

  test('for a failed run', async ($, on) => {
    const error = 'backend failure: three failed calls'
    const chunks = [
      stderr(`run folder: ${RUN_DIR}\n`),
      stdout(`${JSON.stringify({ status: 'error', code: 3, error, run_dir: RUN_DIR })}\n`),
    ]
    const w = world(on, { surfaces: [], chunks, end: { code: 3, signal: null } })
    await boot($)
    expect((await improve($, 'Summarise the meeting notes.')).exitCode).toBe(3)
    expect(w.fills).toEqual([])
    expect(w.reads).toBe(0)
  })

  test('for a cancelled run', async ($, on) => {
    let release = () => {}
    const gate = new Promise<void>((resolve) => {
      release = resolve
    })
    const end = { code: null, signal: 'SIGTERM' }
    const w = world(on, { chunks: [stderr(`run folder: ${RUN_DIR}\n`)], gate, end })
    await boot($)
    await improve($, 'Summarise the meeting notes.')
    await w.waiting
    const cancelling = improve($, 'cancel')
    release()
    expect((await cancelling).text).toBe(`cancelled; resume with /improve --resume ${ID}`)
    await w.results()
    expect(w.fills).toEqual([])
    expect(w.reads).toBe(0)
  })

  test('for a plan (--dry)', async ($, on) => {
    const w = world(on)
    await boot($)
    expect((await improve($, '--dry Summarise the meeting notes.')).exitCode).toBe(0)
    expect(w.fills).toEqual([])
    expect(w.reads).toBe(0)
  })
})

describe('where nothing draws a pane', () => {
  test('the box is filled and the text says so', async ($, on) => {
    const w = world(on, { surfaces: [], chunks: finished(IMPROVED) })
    await boot($)
    const answer = await improve($, 'Summarise the meeting notes.')
    expect(answer.exitCode).toBe(0)
    expect(answer.text).toContain(IN_THE_BOX)
    expect(w.fills).toEqual([{ text: IMPROVED.prompt, mode: 'replace' }])
    expect(w.opens).toEqual([])
  })

  test('a kept draft is said in the text too', async ($, on) => {
    const w = world(on, { surfaces: [], chunks: finished(IMPROVED), draft: 'mine' })
    await boot($)
    const answer = await improve($, 'Summarise the meeting notes.')
    expect(answer.text).toContain(KEPT)
    expect(w.fills).toEqual([])
  })
})
