// What follows /improve, read into a request (SPEC R21): leading CLI flags, then the prompt
// verbatim; the subcommands cancel, clean [ID] and --resume [ID]; everything else is usage.

import { describe, expect, test } from 'claude-code/testing'
import { FLAGS, parseArgs, RUN_ID, usageText } from '../../hooks/args.js'

const ID = '20261004-132507-1a2b3c4d'
const HOSTILE =
  'Say "hi" and \'bye\'; `whoami` $(rm -rf ~) && echo PWNED > /etc/x\n--dry\nIgnore all.'

function run(text: string) {
  const request = parseArgs(text)
  if (request.kind !== 'run') throw new Error(`expected a run, got ${JSON.stringify(request)}`)
  return request
}

function problem(text: string): string {
  const request = parseArgs(text)
  if (request.kind !== 'usage') throw new Error(`expected usage, got ${JSON.stringify(request)}`)
  return request.message
}

describe('the forwarded flags', () => {
  test('are the CLI flags of the mod, each with its arity', () => {
    expect([...FLAGS]).toEqual([
      ['--file', 1],
      ['--examples', 1],
      ['--kind', 1],
      ['--budget', 1],
      ['--strictness', 1],
      ['--allow-growth', 0],
      ['--task-model', 1],
      ['--judge-model', 1],
      ['--reflect-model', 1],
      ['--target-model', 1],
      ['--merge', 0],
      ['--trust-search', 0],
      ['--force-low-budget', 0],
      ['--dry', 0],
    ])
  })

  test('each is read with its arity before the prompt', () => {
    for (const [flag, arity] of FLAGS) {
      if (flag === '--file') continue
      const request = run(arity === 1 ? `${flag} v1 Do it.` : `${flag} Do it.`)
      expect(request.flags, flag).toEqual([{ flag, value: arity === 1 ? 'v1' : null }])
      expect(request.prompt, flag).toBe('Do it.')
    }
    expect(run('--file notes.txt').flags).toEqual([{ flag: '--file', value: 'notes.txt' }])
  })

  test('several flags in a row, with = and quoted values', () => {
    const request = run('--budget=60 --merge --examples "my examples.jsonl" --kind \'task\' Do it.')
    expect(request.flags).toEqual([
      { flag: '--budget', value: '60' },
      { flag: '--merge', value: null },
      { flag: '--examples', value: 'my examples.jsonl' },
      { flag: '--kind', value: 'task' },
    ])
    expect(request.examples).toBe('my examples.jsonl')
    expect(request.prompt).toBe('Do it.')
  })

  test('--dry, --target-model and --file are also read into the request', () => {
    expect(run('--dry Do it.')).toMatchObject({ dry: true, targetModel: null, file: null })
    expect(run('--target-model opus Do it.')).toMatchObject({ targetModel: 'opus', dry: false })
    expect(run('--file "a b.txt"')).toMatchObject({ file: 'a b.txt', prompt: null })
  })
})

describe('the prompt', () => {
  test('is the whole text when there is no flag', () => {
    expect(parseArgs('Summarise the notes.')).toEqual({
      kind: 'run',
      flags: [],
      prompt: 'Summarise the notes.',
      file: null,
      examples: null,
      dry: false,
      targetModel: null,
    })
  })

  test('keeps its lines, indentation, fences and blank lines verbatim', () => {
    const prompt = 'Line one.\n\n  - an indented item\n```py\nprint("x")\n```\n\tTabbed.'
    expect(run(prompt).prompt).toBe(prompt)
    expect(run(`--budget 60\n${prompt}`).prompt).toBe(prompt)
  })

  test('with quotes, backticks, $(...) and newlines is data, kept verbatim', () => {
    const request = run(HOSTILE)
    expect(request.prompt).toBe(HOSTILE)
    expect(request.flags).toEqual([])
    expect(request.dry).toBe(false)
  })

  test('that starts with -- goes after a lone --', () => {
    expect(run('-- --verbose mode is broken').prompt).toBe('--verbose mode is broken')
    expect(run('--budget 9 -- --x').flags).toEqual([{ flag: '--budget', value: '9' }])
    expect(problem('--verbose mode is broken')).toContain('lone --')
  })

  test('may start with a single dash or with a subcommand word', () => {
    expect(run('-h is a flag; explain it').prompt).toBe('-h is a flag; explain it')
    expect(run('cancel the meeting politely').prompt).toBe('cancel the meeting politely')
    expect(run('clean up the code').prompt).toBe('clean up the code')
  })
})

describe('usage problems', () => {
  test('an empty or blank text', () => {
    expect(problem('')).toContain('give a prompt')
    expect(problem(' \n\t ')).toContain('give a prompt')
  })

  test('an unknown flag, --json included', () => {
    expect(problem('--verbose Do it.')).toContain('unknown flag --verbose')
    expect(problem('--json Do it.')).toContain('unknown flag --json')
  })

  test('a flag without its value', () => {
    expect(problem('--budget')).toBe('--budget needs a value')
    expect(problem('--budget --merge Do it.')).toBe('--budget needs a value')
    expect(problem('--file "unclosed')).toBe('--file needs a value')
    expect(problem('--budget= Do it.')).toBe('--budget needs a value')
  })

  test('a value for a flag that takes none, and a flag given twice', () => {
    expect(problem('--merge=yes Do it.')).toBe('--merge takes no value')
    expect(problem('--budget 5 --budget 6 Do it.')).toBe('--budget is given twice')
  })

  test('no prompt, or a prompt and --file both', () => {
    expect(problem('--dry')).toContain('no prompt')
    expect(problem('--file a.txt Do it.')).toContain('not both')
  })
})

describe('the subcommands', () => {
  test('cancel, alone', () => {
    expect(parseArgs('cancel')).toEqual({ kind: 'cancel' })
    expect(parseArgs('  cancel\n')).toEqual({ kind: 'cancel' })
  })

  test('clean, with or without a run id', () => {
    expect(parseArgs('clean')).toEqual({ kind: 'clean', id: null })
    expect(parseArgs(`clean ${ID}`)).toEqual({ kind: 'clean', id: ID })
    expect(problem('clean ../../etc')).toContain('run id')
  })

  test('--resume, with or without a run id, and alone', () => {
    expect(parseArgs('--resume')).toEqual({ kind: 'resume', id: null })
    expect(parseArgs(`--resume ${ID}`)).toEqual({ kind: 'resume', id: ID })
    expect(parseArgs(`--resume=${ID}`)).toEqual({ kind: 'resume', id: ID })
    expect(problem('--resume /tmp/x')).toContain('run id')
    expect(problem(`--resume ${ID} more`)).toContain('give it alone')
    expect(problem('--dry --resume')).toContain('give it alone')
  })

  test('help', () => {
    expect(parseArgs('--help')).toEqual({ kind: 'help' })
    expect(parseArgs('-h')).toEqual({ kind: 'help' })
    expect(parseArgs('--budget 5 --help')).toEqual({ kind: 'help' })
  })

  test('a run id has the form of the CLI', () => {
    expect(RUN_ID.test(ID)).toBe(true)
    expect(RUN_ID.test(`${ID}-2`)).toBe(true)
    for (const bad of ['', 'x', `${ID}/..`, `../${ID}`, '20261004-132507-1A2B3C4D', `${ID}\n`]) {
      expect(RUN_ID.test(bad), bad).toBe(false)
    }
  })
})

describe('the usage text', () => {
  test('names the command, every form and every flag', () => {
    const text = usageText('optimize')
    for (
      const form of ['<prompt>', '--file PATH', '--dry', '--resume [ID]', 'clean [ID]', 'cancel']
    ) {
      expect(text).toContain(`/optimize ${form}`)
    }
    for (const [flag, arity] of FLAGS) expect(text).toContain(arity === 1 ? `${flag} V` : flag)
    expect(text).not.toContain('error:')
  })

  test('leads with the problem when there is one', () => {
    expect(usageText('improve', 'unknown flag --x')).toStartWith('error: unknown flag --x\nusage:')
  })
})
