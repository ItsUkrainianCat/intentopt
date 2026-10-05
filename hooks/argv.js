// The command lines and the environment of the child processes the mod starts (SPEC R19, R21;
// ADR-010): `uv run --frozen --project <plugin root> autoimprover --json ...` as an argument
// list, never a shell line; the prompt only as a file path; the virtual environment outside the
// plugin cache. Pure: the caller passes in what it read from the engine.

// A prompt longer than this many characters gets a `--dry` run first (SPEC R21).
export const DRY_FIRST_CHARS = 2000

/** @import { Flag } from './args.js' */
/**
 * @typedef {{ flags: Flag[], prompt: string | null, file: string | null, dry: boolean,
 *   targetModel: string | null }} RunRequest
 */

/**
 * The whole command line of one call of the CLI: `args` after the tool's name and `--json`.
 * @param {string} root the plugin's folder, where `pyproject.toml` and `uv.lock` are
 * @param {readonly string[]} args
 * @returns {string[]}
 */
export function cliArgv(root, args) {
  return ['uv', 'run', '--frozen', '--project', root, 'autoimprover', '--json', ...args]
}

/**
 * The CLI's arguments for a run (or with `dry`, its plan): the flags as given, the paths made
 * absolute, the prompt as `--file <promptFile>` when the person typed it, and the session's
 * model as `--target-model` unless they chose one (SPEC R14a, R21).
 * @param {RunRequest} request
 * @param {{ promptFile: string | null, paths: Map<string, string>, targetModel: string | null,
 *   dry: boolean }} given `paths` maps a flag (`--file`, `--examples`) to its absolute path
 * @returns {string[]}
 */
export function runArgs(request, given) {
  const args = []
  for (const { flag, value } of request.flags) {
    if (flag === '--dry' || flag === '--file') continue
    args.push(flag)
    if (value !== null) args.push(given.paths.get(flag) ?? value)
  }
  const file = given.promptFile ?? given.paths.get('--file') ?? request.file
  if (file !== null) args.push('--file', file)
  if (request.targetModel === null && given.targetModel !== null) {
    args.push('--target-model', given.targetModel)
  }
  if (given.dry) args.push('--dry')
  return args
}

/**
 * @param {string} id
 * @returns {string[]}
 */
export function resumeArgs(id) {
  return ['--resume', id]
}

/**
 * @param {string} id
 * @returns {string[]}
 */
export function cleanArgs(id) {
  return ['clean', id]
}

/**
 * Whether a run starts with its plan: a prompt longer than DRY_FIRST_CHARS characters (code
 * points, as the CLI counts them), or a `--file` of more bytes than that, since a file's
 * characters are not read here.
 * @param {RunRequest} request
 * @param {number | null} fileBytes the size of the `--file`, when there is one
 * @returns {boolean}
 */
export function needsDryFirst(request, fileBytes) {
  if (request.dry) return false
  if (request.prompt !== null) return [...request.prompt].length > DRY_FIRST_CHARS
  return fileBytes !== null && fileBytes > DRY_FIRST_CHARS
}

/**
 * The child's environment over the engine's: uv keeps the tool's virtual environment in the
 * person's cache folder, never inside the plugin's (installed plugins are replaced on update).
 * @param {string} venv
 * @returns {Record<string, string>}
 */
export function childEnv(venv) {
  return { UV_PROJECT_ENVIRONMENT: venv }
}

/**
 * `$XDG_CACHE_HOME/autoimprover/venv`, else `$HOME/.cache/autoimprover/venv`; a relative value
 * is ignored, as the XDG base directory specification asks. Null when neither is usable.
 * @param {{ home: string | undefined, cacheHome: string | undefined }} env
 * @returns {string | null}
 */
export function venvFolder(env) {
  const cache = isAbsolute(env.cacheHome)
    ? trimSlash(env.cacheHome)
    : isAbsolute(env.home)
    ? `${trimSlash(env.home)}/.cache`
    : null
  return cache === null ? null : `${cache}/autoimprover/venv`
}

/**
 * The `mktemp -d` template of the private prompt folder: under `$TMPDIR` when it is absolute,
 * else under /tmp.
 * @param {string | undefined} tmpdir
 * @returns {string}
 */
export function tmpTemplate(tmpdir) {
  const base = isAbsolute(tmpdir) ? trimSlash(tmpdir) : '/tmp'
  return `${base}/autoimprover-prompt.XXXXXX`
}

/**
 * `path` as the CLI should read it: absolute paths as they are, `~/...` under `home`, the rest
 * under the session's folder (the child runs in its own empty folder).
 * @param {string} path
 * @param {{ cwd: string, home: string | undefined }} where
 * @returns {string}
 */
export function resolvePath(path, where) {
  if (path.startsWith('/')) return path
  if (path.startsWith('~/') && isAbsolute(where.home)) {
    return `${trimSlash(where.home)}/${path.slice(2)}`
  }
  return `${trimSlash(where.cwd)}/${path}`
}

/**
 * The session's model as `--target-model` takes it (a model id or alias, maybe with a `[1m]`
 * style suffix); null when it does not look like one (a display name).
 * @param {string} model
 * @returns {string | null}
 */
export function targetModelOf(model) {
  const name = model.trim()
  return /^[A-Za-z0-9][A-Za-z0-9._:/-]*(?:\[[^\]\s]*\])?$/.test(name) ? name : null
}

/**
 * @param {string | undefined} path
 * @returns {path is string}
 */
function isAbsolute(path) {
  return path !== undefined && path.startsWith('/')
}

/**
 * @param {string} path
 * @returns {string}
 */
function trimSlash(path) {
  return path.length > 1 ? path.replace(/\/+$/, '') : path
}
