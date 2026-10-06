// What follows `/improve` (or `/optimize`), read into a request (SPEC R21): leading flags of the
// CLI, then the prompt, verbatim, as the rest of the text; or one of the subcommands `cancel`,
// `clean [ID]` and `--resume [ID]`. Pure: no engine, no I/O. The prompt never becomes a command
// line argument; the caller writes it to a file (SPEC R19).

// Every flag of the CLI (src/autoimprover/cli.py, `_parser`) the mod forwards, with the number of
// values it takes. `--help`, `--json` (always set) and `--resume` (a subcommand) are not in it.
export const FLAGS = new Map([
  ['--file', 1],
  ['--examples', 1],
  ['--kind', 1],
  ['--time', 1],
  ['--deep', 0],
  ['--workers', 1],
  ['--budget', 1],
  ['--strictness', 1],
  ['--allow-growth', 0],
  ['--task-model', 1],
  ['--judge-model', 1],
  ['--reflect-model', 1],
  ['--target-model', 1],
  ['--effort', 1],
  ['--task-effort', 1],
  ['--judge-effort', 1],
  ['--reflect-effort', 1],
  ['--merge', 0],
  ['--trust-search', 0],
  ['--force-low-budget', 0],
  ['--dry', 0],
])

// A run id, the only form `--resume` and `clean` take (src/autoimprover/types.py RUN_ID_PATTERN).
export const RUN_ID = /^\d{8}-\d{6}-[0-9a-f]{8}(?:-\d+)?$/

/**
 * @typedef {{ flag: string, value: string | null }} Flag
 * @typedef {{ kind: 'usage', message: string }
 *   | { kind: 'help' }
 *   | { kind: 'cancel' }
 *   | { kind: 'clean', id: string | null }
 *   | { kind: 'resume', id: string | null }
 *   | { kind: 'run', flags: Flag[], prompt: string | null, file: string | null,
 *       examples: string | null, dry: boolean, targetModel: string | null }} Request
 */

/**
 * The request in `text`, everything after the command's name as typed (it may span lines).
 * @param {string} text
 * @returns {Request}
 */
export function parseArgs(text) {
  const whole = text.trim()
  if (whole === '') {
    return usage('give a prompt, --file PATH, --resume [ID], clean [ID] or cancel')
  }
  if (whole === 'cancel') return { kind: 'cancel' }
  if (whole === '-h' || whole === '--help') return { kind: 'help' }
  const clean = /^clean(?:\s+(\S+))?$/.exec(whole)
  if (clean !== null) {
    const id = clean[1] ?? null
    if (id !== null && !RUN_ID.test(id)) return usage(`clean takes a run id, not ${id}`)
    return { kind: 'clean', id }
  }
  return parseRun(text)
}

/**
 * @param {string} text
 * @returns {Request}
 */
function parseRun(text) {
  /** @type {Flag[]} */
  const flags = []
  let rest = text.trimStart()
  while (rest.startsWith('--')) {
    const token = /^\S+/.exec(rest)?.[0] ?? ''
    rest = rest.slice(token.length)
    if (token === '--') {
      rest = rest.trimStart()
      break
    }
    const [flag, inline] = splitInline(token)
    if (flag === '--help') return { kind: 'help' }
    if (flag === '--resume') return parseResume(flags, inline === null ? rest : `${inline}${rest}`)
    const arity = FLAGS.get(flag)
    if (arity === undefined) {
      return usage(`unknown flag ${flag}; a prompt that starts with -- goes after a lone --`)
    }
    if (flags.some((given) => given.flag === flag)) return usage(`${flag} is given twice`)
    if (arity === 0) {
      if (inline !== null) return usage(`${flag} takes no value`)
      flags.push({ flag, value: null })
      rest = rest.trimStart()
      continue
    }
    const read = inline !== null ? { value: inline, rest } : readValue(rest)
    if (read === null || read.value === '' || read.value.startsWith('--')) {
      return usage(`${flag} needs a value`)
    }
    flags.push({ flag, value: read.value })
    rest = read.rest.trimStart()
  }
  const value = (/** @type {string} */ flag) => flags.find((f) => f.flag === flag)?.value ?? null
  const prompt = rest === '' ? null : rest
  const file = value('--file')
  if (prompt !== null && file !== null) {
    return usage('give the prompt as text or with --file, not both')
  }
  if (prompt === null && file === null) {
    return usage('no prompt: give it after the flags, or with --file PATH')
  }
  return {
    kind: 'run',
    flags,
    prompt,
    file,
    examples: value('--examples'),
    dry: flags.some((f) => f.flag === '--dry'),
    targetModel: value('--target-model'),
  }
}

/**
 * `--resume` alone (the last run) or with one run id, and nothing else.
 * @param {Flag[]} flags
 * @param {string} rest
 * @returns {Request}
 */
function parseResume(flags, rest) {
  const id = rest.trim()
  if (flags.length > 0 || /\s/.test(id)) {
    return usage('--resume continues a saved run with its own prompt and flags; give it alone')
  }
  if (id !== '' && !RUN_ID.test(id)) return usage(`--resume takes a run id, not ${id}`)
  return { kind: 'resume', id: id === '' ? null : id }
}

/**
 * `--flag=value` as [flag, value]; [token, null] without `=`.
 * @param {string} token
 * @returns {[string, string | null]}
 */
function splitInline(token) {
  const at = token.indexOf('=')
  return at < 0 ? [token, null] : [token.slice(0, at), token.slice(at + 1)]
}

/**
 * A flag's value at the start of `rest`: one word, or text in double or single quotes (no
 * escapes), so a path may hold spaces; null when it is missing or its quote never closes.
 * @param {string} rest
 * @returns {{ value: string, rest: string } | null}
 */
function readValue(rest) {
  const text = rest.trimStart()
  const quote = text[0]
  if (quote === '"' || quote === "'") {
    const end = text.indexOf(quote, 1)
    if (end < 0) return null
    return { value: text.slice(1, end), rest: text.slice(end + 1) }
  }
  const word = /^\S+/.exec(text)?.[0]
  if (word === undefined) return null
  return { value: word, rest: text.slice(word.length) }
}

/**
 * @param {string} message
 * @returns {Request}
 */
function usage(message) {
  return { kind: 'usage', message }
}

/**
 * The usage text of `/<command>`, after `problem` when there is one.
 * @param {string} command
 * @param {string} [problem]
 * @returns {string}
 */
export function usageText(command, problem) {
  const flags = [...FLAGS].map(([flag, arity]) => (arity === 1 ? `${flag} V` : flag))
  const lines = [
    `/${command} <prompt>                improve the prompt (it may span lines)`,
    `/${command} [flags] <prompt>        a prompt that starts with -- goes after a lone --`,
    `/${command} --file PATH [flags]     the prompt from a file`,
    `/${command} --dry <prompt>          the plan only: no model call, nothing written`,
    `/${command} --resume [ID]           continue the last run, or run ID`,
    `/${command} clean [ID]              remove the last run's folder, or run ID's`,
    `/${command} cancel                  stop the running run (it stays resumable)`,
    `flags: ${flags.join(', ')}`,
  ]
  return [...(problem === undefined ? [] : [`error: ${problem}`]), 'usage:', ...lines].join('\n')
}
