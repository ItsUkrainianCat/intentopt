// The autoimprover mod (SPEC R19, R21; ADR-010). `/improve` and its alias `/optimize` run the
// autoimprover CLI as a child process: by an argument list (no shell), with a typed prompt in a
// file inside a private folder made by `mktemp -d`, the session's model as the target model, and
// the tool's virtual environment in the person's cache folder. The child's stderr feeds the
// status line and its one JSON object the report pane; an improved prompt goes into an empty
// prompt box by itself as an editable draft (a typed draft is kept; "Use it" replaces it), and
// nothing is sent until the person presses Enter. The model never sees the prompt:
// the module hooks `session.start`, `command.run` and the drawing of its own pane, nothing else.
// Every call on `$` is in this file; parsing, command lines and the report view are the pure
// modules beside it.

import { RUN_ID, parseArgs, usageText } from './args.js'
import { boxPlan, boxRefused } from './box.js'
import {
  childEnv, cleanArgs, cliArgv, needsDryFirst, resolvePath, resumeArgs, runArgs, targetModelOf,
  tmpTemplate, venvFolder,
} from './argv.js'
import { paneTree } from './pane.js'
import {
  cancelledText, cleanedText, endOf, errorView, lastLines, messageOf, oneLine, parseObject,
  renderText, runIdOf, runningView, viewOf,
} from './report.js'
import { Progress } from './stream.js'

/**
 * @import { CommandRunResult, EngineInterface as E, HookStream, On } from 'claude-code'
 * @import { ProcessSpawnChunk, ProcessSpawnResult } from 'claude-code'
 * @import { RenderElement, RenderSurface, UiPressArgument } from 'claude-code'
 */
/** @import { Request } from './args.js' */
/** @import { View } from './report.js' */
/**
 * One run of the CLI, from `/improve` to the end of its child.
 * @typedef {{ command: string, cancelled: boolean, detached: boolean, tmp: string | null,
 *   runDir: string | null, stream: HookStream<ProcessSpawnChunk, ProcessSpawnResult> | null
 * }} Job
 * @typedef {{ env: Record<string, string>, home: string | undefined }} Setup
 */

const ABOUT = 'Improve a prompt with autoimprover (scored by running it); you review the result'
const ALIAS = 'Alias of /improve'
const HINT = '<prompt> | --file PATH | --dry <prompt> | --resume [ID] | clean [ID] | cancel'
// Both commands run at once when typed mid-turn, so `cancel` can stop a run while Claude works.
const COMMAND = { argumentHint: HINT, immediate: /** @type {const} */ (true) }
const FOCUS = /** @type {const} */ (true)
// `uv run` may build the tool's environment first, so a plan or a clean may take this long.
const ONE_SHOT_MS = 300_000

/** The run in progress; one per session, so `cancel` knows which. @type {Job | null} */
let current = null
/** What the pane draws: the run in progress, or how the last one ended. @type {View | null} */
let shown = null

/** A stop before the child starts; `result` is the command's answer. */
class Refusal extends Error {
  /** @param {string} text @param {number} exitCode */
  constructor(text, exitCode) {
    super(text)
    this.result = { text, exitCode }
  }
}

/** @param {On} on */
export function register(on) {
  on('session.start', startSession)
  on('command.run', { command: ['improve', 'optimize'] }, runCommand)
  on('ui.render', { component: 'Pane', requestId: 'improve' }, drawPane)
}

/**
 * Lets the session start, then registers the commands; a name another plugin holds is logged and
 * does not stop the session.
 * @param {E} $ @param {any} e @param {(e: any) => Promise<any>} next
 */
async function startSession($, e, next) {
  const started = await next(e)
  try {
    await $.command.register({ name: 'improve', description: ABOUT, ...COMMAND })
  } catch (error) {
    $.ui.log(`autoimprover: /improve is not available: ${messageOf(error)}`)
  }
  try {
    await $.command.register({ name: 'optimize', description: ALIAS, ...COMMAND })
  } catch (error) {
    $.ui.log(`autoimprover: /optimize is not available: ${messageOf(error)}`)
  }
  return started
}

/**
 * `/improve ...` and `/optimize ...`, the boundary of every command: a refusal is the answer, any
 * other failure is reported as exit 1 with its message rather than as a silent hook failure.
 * @param {E} $ @param {{ command: string, args: string }} e
 * @returns {Promise<CommandRunResult>}
 */
async function runCommand($, e) {
  const { command } = e
  const request = parseArgs(e.args)
  try {
    if (request.kind === 'usage') return { text: usageText(command, request.message), exitCode: 2 }
    if (request.kind === 'help') return { text: usageText(command) }
    if (request.kind === 'cancel') return await cancelRun(command)
    if (request.kind === 'clean') return await cleanRun($, command, request.id)
    if (current !== null) {
      const id = runIdOf(current.runDir)
      const which = id === null ? 'a run is already going' : `run ${id} is still going`
      return { text: `${which}; /${command} cancel stops it`, exitCode: 2 }
    }
    return await startRun($, command, request)
  } catch (error) {
    if (error instanceof Refusal) return error.result
    return { text: `autoimprover: failed: ${messageOf(error)}`, exitCode: 1 }
  }
}

/**
 * Ends the run in progress by ending the loop over its child, which kills that child by the
 * engine's own handle (SPEC R21); the run folder stays, so the run can be resumed.
 * @param {string} command
 * @returns {Promise<CommandRunResult>}
 */
async function cancelRun(command) {
  const job = current
  if (job === null) return { text: 'no autoimprover run is going' }
  job.cancelled = true
  if (job.stream !== null) await job.stream.return({ code: null, signal: null })
  return { text: cancelledText(runIdOf(job.runDir), command) }
}

/**
 * A run or a resume: the checks, the private folder and the prompt file, the plan when one is
 * due, then the child: detached where a pane can show how it ends, else awaited (a `-p` run) with
 * the report as the command's text. The folder is removed however the child ends.
 * @param {E} $ @param {string} command @param {Request & { kind: 'run' | 'resume' }} request
 * @returns {Promise<CommandRunResult>}
 */
async function startRun($, command, request) {
  /** @type {Job} */
  const job = { command, cancelled: false, detached: false, tmp: null, runDir: null, stream: null }
  current = job
  try {
    const setup = await prepare($)
    const tmp = (job.tmp = await makeFolder($))
    let planned = ''
    let args
    if (request.kind === 'resume') {
      args = resumeArgs(await runIdFor($, request.id, `/${command} --resume ID`))
    } else {
      const prompt = request.prompt
      const promptFile = prompt === null ? null : await writePrompt($, tmp, prompt)
      const paths = await resolvePaths($, request, setup.home)
      const given = { promptFile, paths, targetModel: await targetModel($, request) }
      const dryArgs = runArgs(request, { ...given, dry: true })
      if (request.dry) {
        const view = await plan($, job, setup, dryArgs)
        return { text: renderText(view), exitCode: view.state === 'plan' ? 0 : view.exitCode }
      }
      if (needsDryFirst(request, await fileSize($, paths.get('--file')))) {
        const view = await plan($, job, setup, dryArgs)
        if (view.exitCode !== 0) return { text: renderText(view), exitCode: view.exitCode }
        planned = `${renderText(view)}\n`
      }
      args = runArgs(request, { ...given, dry: false })
    }
    if (job.cancelled) return { text: cancelledText(null, command), exitCode: 130 }
    const argv = cliArgv($.plugin.root, args)
    if ((await $.session.surfaces()).length === 0) {
      const view = await fillBox($, await drive($, job, argv, setup.env))
      return { text: renderText(view), exitCode: view.exitCode }
    }
    shown = runningView({ status: null, runDir: null, command })
    await openPane($, false)
    job.detached = true
    void runDetached($, job, argv, setup.env)
    const started = 'autoimprover is running; its report opens in a pane.'
    return { text: `${planned}${started} /${command} cancel stops it.` }
  } finally {
    if (!job.detached) await finishJob($, job)
  }
}

/**
 * `uv` on the PATH and a folder for the tool's virtual environment, or a refusal saying which.
 * @param {E} $
 * @returns {Promise<Setup>}
 */
async function prepare($) {
  const uv = await $.process.run(['uv', '--version']).catch((error) => {
    const why = `uv did not start (${messageOf(error)})`
    throw new Refusal(`autoimprover needs uv: ${why}; put it on the PATH Claude Code runs with`, 2)
  })
  if (uv.exitCode !== 0) {
    throw new Refusal(`autoimprover needs uv; \`uv --version\` failed: ${lastLines(uv.stderr)}`, 2)
  }
  const home = await $.env.get('HOME')
  const venv = venvFolder({ home, cacheHome: await $.env.get('XDG_CACHE_HOME') })
  if (venv === null) {
    throw new Refusal('HOME is not an absolute path: there is no cache folder for the tool', 2)
  }
  return { env: childEnv(venv), home }
}

/**
 * A private folder made by `mktemp -d` (mode 0700): the prompt file's, and the child's cwd.
 * @param {E} $
 * @returns {Promise<string>}
 */
async function makeFolder($) {
  const made = await $.process.run(['mktemp', '-d', tmpTemplate(await $.env.get('TMPDIR'))])
  const folder = made.stdout.trim()
  if (made.exitCode !== 0 || !folder.startsWith('/')) {
    throw new Error(`mktemp -d failed: ${lastLines(made.stderr)}`)
  }
  return folder
}

/**
 * The typed prompt as `<folder>/prompt.txt`, mode 0600 (SPEC R21): it reaches the CLI as a path,
 * never as an argument or a shell line (SPEC R19).
 * @param {E} $ @param {string} folder @param {string} prompt
 * @returns {Promise<string>}
 */
async function writePrompt($, folder, prompt) {
  const file = `${folder}/prompt.txt`
  await $.fs.write(file, prompt)
  const chmod = await $.process.run(['chmod', '600', '--', file])
  if (chmod.exitCode !== 0) {
    throw new Error(`chmod 600 ${file} failed: ${lastLines(chmod.stderr)}`)
  }
  return file
}

/**
 * `--file` and `--examples` as absolute paths, since the child runs in its own folder.
 * @param {E} $ @param {Request & { kind: 'run' }} request @param {string | undefined} home
 * @returns {Promise<Map<string, string>>}
 */
async function resolvePaths($, request, home) {
  const paths = new Map()
  const given = request.flags.filter((f) => f.flag === '--file' || f.flag === '--examples')
  if (given.length === 0) return paths
  const cwd = await $.session.cwd()
  for (const { flag, value } of given) paths.set(flag, resolvePath(value ?? '', { cwd, home }))
  return paths
}

/**
 * The session's model as `--target-model` (SPEC R14a, R21), unless the person chose one.
 * @param {E} $ @param {Request & { kind: 'run' }} request
 * @returns {Promise<string | null>}
 */
async function targetModel($, request) {
  if (request.targetModel !== null) return null
  const model = await $.session.model()
  const target = targetModelOf(model)
  if (target !== null) return target
  const said = `the session's model "${oneLine(model)}" is not a model id`
  throw new Refusal(`${said}; pass --target-model`, 2)
}

/**
 * The size in bytes of the `--file`, when there is one.
 * @param {E} $ @param {string | undefined} file
 * @returns {Promise<number | null>}
 */
async function fileSize($, file) {
  if (file === undefined) return null
  const stat = await $.fs.stat(file).catch((error) => {
    throw new Refusal(`--file: cannot read ${file}: ${messageOf(error)}`, 2)
  })
  return stat.size
}

/**
 * The given run id, or the last run's (kept in the store), or a refusal showing `form`.
 * @param {E} $ @param {string | null} id @param {string} form
 * @returns {Promise<string>}
 */
async function runIdFor($, id, form) {
  if (id !== null) return id
  const last = await $.store.get('lastRunId')
  if (typeof last === 'string' && RUN_ID.test(last)) return last
  throw new Refusal(`no last run is known; give its id: ${form}`, 2)
}

/**
 * The plan of `--dry` (SPEC R4): one short child, no model call, nothing written.
 * @param {E} $ @param {Job} job @param {Setup} setup @param {string[]} args
 * @returns {Promise<View>}
 */
async function plan($, job, setup, args) {
  const init = { cwd: job.tmp ?? undefined, env: setup.env, timeoutMs: ONE_SHOT_MS }
  const done = await $.process.run(cliArgv($.plugin.root, args), init)
  return viewOf(endOf(done, job.command))
}

/**
 * `/improve clean [ID]`: one run folder removed through the CLI, the last run's by default.
 * @param {E} $ @param {string} command @param {string | null} id
 * @returns {Promise<CommandRunResult>}
 */
async function cleanRun($, command, id) {
  const setup = await prepare($)
  const target = await runIdFor($, id, `/${command} clean ID`)
  const init = { env: setup.env, timeoutMs: ONE_SHOT_MS }
  const done = await $.process.run(cliArgv($.plugin.root, cleanArgs(target)), init)
  const result = parseObject(done.stdout)
  if (result === null || result.status !== 'cleaned') {
    const view = viewOf(endOf(done, command))
    return { text: renderText(view), exitCode: view.exitCode }
  }
  if ((await $.store.get('lastRunId')) === target) await $.store.delete('lastRunId')
  return { text: cleanedText(result, target) }
}

/**
 * Runs the child to its end: stdout collected, each stderr line to the status line, the run
 * folder caught and its id kept for `--resume` as soon as the CLI prints it (SPEC R21).
 * @param {E} $ @param {Job} job @param {string[]} argv @param {Record<string, string>} env
 * @returns {Promise<View>}
 */
async function drive($, job, argv, env) {
  const progress = new Progress()
  const stream = $.process.spawn({ argv, env, cwd: job.tmp ?? undefined })
  job.stream = stream
  try {
    for await (const chunk of stream) await take($, job, progress.push(chunk))
  } catch (error) {
    if (!job.cancelled) throw error // a cancelled stream may end by throwing: that is the cancel
  }
  await take($, job, progress.finish())
  /** @type {ProcessSpawnResult | null} */
  let end = null
  try {
    end = await stream.result
  } catch (error) {
    if (!job.cancelled) throw error // closed before its end: only a cancel does that
  }
  const { stdout, stderr } = progress
  const { command, cancelled, runDir } = job
  const [code, signal] = [end?.code ?? null, end?.signal ?? null]
  return viewOf({ stdout, stderr, code, signal, cancelled, runDir, command })
}

/**
 * What one chunk changed: the run folder (kept as the last run), the status line, the pane.
 * @param {E} $ @param {Job} job @param {{ status: string | null, runDir: string | null }} seen
 */
async function take($, job, seen) {
  if (seen.runDir !== null) {
    job.runDir = seen.runDir
    const id = runIdOf(seen.runDir)
    if (id !== null && RUN_ID.test(id)) await $.store.set('lastRunId', id)
  }
  if (seen.status === null) return
  $.ui.status(`autoimprover: ${seen.status}`)
  if (job.detached) {
    shown = runningView({ status: seen.status, runDir: job.runDir, command: job.command })
    $.ui.invalidate('ui.render')
  }
}

/**
 * The detached loop of an interactive run. It has no caller, so a failure becomes its report.
 * @param {E} $ @param {Job} job @param {string[]} argv @param {Record<string, string>} env
 */
async function runDetached($, job, argv, env) {
  /** @type {View} */
  let view
  try {
    view = await drive($, job, argv, env)
  } catch (error) {
    const { command, cancelled, runDir } = job
    const end = { stdout: '', stderr: '', code: null, signal: null, cancelled, runDir, command }
    view = errorView(1, messageOf(error), end)
  } finally {
    await finishJob($, job)
  }
  shown = view = await fillBox($, view)
  $.ui.invalidate('ui.render')
  $.ui.toast(view.toast)
  if (!(await openPane($, true))) $.ui.log(renderText(view))
}

/**
 * An improved prompt into the prompt box by itself, unless a draft the person typed is there
 * (SPEC R21); the view then says which happened. Nothing is sent.
 * @param {E} $ @param {View} view
 * @returns {Promise<View>}
 */
async function fillBox($, view) {
  const improved = view.improved
  if (improved === null) return view
  let placing = boxPlan(view, (await $.prompt.read()).text) ?? boxRefused(view, undefined)
  if (placing.fill) {
    const filled = await $.prompt.fill({ text: improved, mode: 'replace' })
    if (!filled.isFilled) placing = boxRefused(view, filled.refusal)
  }
  return { ...view, box: placing.box, toast: placing.toast }
}

/**
 * However the run ended: it is over, the status line is cleared, the prompt's folder is removed.
 * @param {E} $ @param {Job} job
 */
async function finishJob($, job) {
  if (current === job) current = null
  $.ui.status(undefined)
  if (job.tmp === null) return
  let said = ''
  try {
    const removed = await $.process.run(['rm', '-rf', '--', job.tmp])
    if (removed.exitCode !== 0) said = lastLines(removed.stderr)
  } catch (error) {
    said = messageOf(error)
  }
  if (said !== '') $.ui.log(`autoimprover: could not remove ${job.tmp}: ${said}`)
}

/**
 * Opens the pane, with the keyboard and Escape to close it once there is a result; false when
 * it is not drawn now, so the caller falls back to text.
 * @param {E} $ @param {boolean} result
 * @returns {Promise<boolean>}
 */
async function openPane($, result) {
  try {
    const asks = result ? { focus: FOCUS, closeOnEscape: FOCUS } : {}
    return (await $.ui.open({ id: 'improve', title: 'improve', ...asks })).isPlaced
  } catch (error) {
    $.ui.log(`autoimprover: the report pane did not open: ${messageOf(error)}`)
    return false
  }
}

/**
 * The pane: what `shown` holds, drawn with the surface's elements, and its buttons.
 * @param {E} $ @param {any} e
 * @returns {RenderElement}
 */
function drawPane($, e) {
  const elements = $.ui.resolve(e)
  return paneTree(shown, elements, shown === null ? [] : buttonsOf($, elements.Button, shown))
}

/**
 * "Use it" (puts the improved prompt into the box again, over a draft), Copy and Close for an
 * improved prompt; Close (Hide while it runs) otherwise.
 * @param {E} $ @param {any} Button the surface's Button element @param {View} view
 */
function buttonsOf($, Button, view) {
  const prompt = view.improved
  const close = () => closePane($)
  const dismiss = { role: 'dismiss' }
  if (prompt === null || view.useLabel === null) {
    const label = view.state === 'running' ? 'Hide' : 'Close'
    return [h(Button, { key: 'close', label, hotkey: 'x', ...dismiss, onPress: close })]
  }
  const use = () => useImproved($, prompt)
  /** @param {UiPressArgument} press */
  const copy = (press) => copyImproved($, prompt, press.surface)
  const main = { variant: 'primary', autoFocus: true }
  return [
    h(Button, { key: 'use', label: view.useLabel, hotkey: 'u', ...main, onPress: use }),
    h(Button, { key: 'copy', label: 'Copy', hotkey: 'c', onPress: copy }),
    h(Button, { key: 'close', label: 'Close', hotkey: 'x', ...dismiss, onPress: close }),
  ]
}

/**
 * Puts the improved prompt into the prompt box as an editable draft, then closes the pane.
 * @param {E} $ @param {string} prompt
 */
async function useImproved($, prompt) {
  const filled = await $.prompt.fill({ text: prompt, mode: 'replace' })
  if (!filled.isFilled) {
    const why = filled.refusal ?? 'refused'
    $.ui.toast(`autoimprover: the prompt box did not take it (${why}); use Copy`)
    return
  }
  $.ui.toast('autoimprover: the improved prompt is in the prompt box; edit it or send it')
  await $.ui.close({ id: 'improve' })
}

/** @param {E} $ @param {string} prompt @param {RenderSurface} surface */
async function copyImproved($, prompt, surface) {
  const copied = await $.ui.copy({ text: prompt, surface })
  const said = copied.isCopied ? 'the prompt is on the clipboard' : `not copied (${copied.reason})`
  $.ui.toast(`autoimprover: ${said}`)
}

/** @param {E} $ */
async function closePane($) {
  await $.ui.close({ id: 'improve' })
}
