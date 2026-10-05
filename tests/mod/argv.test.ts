// The command lines and environment of the children (SPEC R19, R21; ADR-010): an argument list
// with the prompt only as a file path, the session's model as the default target, the virtual
// environment outside the plugin's folder, and a dry run first above 2,000 characters.

import { describe, expect, test } from 'claude-code/testing'
import { parseArgs } from '../../hooks/args.js'
import {
  childEnv,
  cleanArgs,
  cliArgv,
  DRY_FIRST_CHARS,
  needsDryFirst,
  resolvePath,
  resumeArgs,
  runArgs,
  targetModelOf,
  tmpTemplate,
  venvFolder,
} from '../../hooks/argv.js'

const ROOT = '/h/u/.claude/plugins/cache/optimizer/autoimprover/0.2.0.dev0'
const FILE = '/t/autoimprover-prompt.abc123/prompt.txt'
const MODEL = 'claude-opus-5-5'
const HOSTILE = 'Say "hi" `whoami` $(rm -rf ~) ; echo PWNED\n--dry\n--target-model x\nIgnore all.'

function run(text: string) {
  const request = parseArgs(text)
  if (request.kind !== 'run') throw new Error(`expected a run, got ${JSON.stringify(request)}`)
  return request
}

const typed = (text: string, dry = false, paths = new Map<string, string>()) =>
  runArgs(run(text), { promptFile: FILE, paths, targetModel: MODEL, dry })

describe('the command line', () => {
  test('is uv run on the plugin folder, the tool, --json, then the arguments', () => {
    expect(cliArgv(ROOT, ['a', 'b'])).toEqual([
      'uv',
      'run',
      '--frozen',
      '--project',
      ROOT,
      'autoimprover',
      '--json',
      'a',
      'b',
    ])
  })

  test('a typed prompt goes as --file, with the session model as the target', () => {
    expect(typed('Summarise the notes.')).toEqual(['--file', FILE, '--target-model', MODEL])
  })

  test('flags are forwarded in order, values as given', () => {
    expect(typed('--budget 60 --merge --strictness bold Do it.')).toEqual([
      '--budget',
      '60',
      '--merge',
      '--strictness',
      'bold',
      '--file',
      FILE,
      '--target-model',
      MODEL,
    ])
  })

  test('a target model the person chose replaces the session model', () => {
    const args = runArgs(run('--target-model sonnet Do it.'), {
      promptFile: FILE,
      paths: new Map(),
      targetModel: null,
      dry: false,
    })
    expect(args).toEqual(['--target-model', 'sonnet', '--file', FILE])
    expect(args.filter((a) => a === '--target-model')).toHaveLength(1)
  })

  test('--file and --examples go as the absolute paths resolved for them', () => {
    const paths = new Map([
      ['--file', '/w/notes.txt'],
      ['--examples', '/w/ex.jsonl'],
    ])
    const args = runArgs(run('--examples ex.jsonl --file notes.txt'), {
      promptFile: null,
      paths,
      targetModel: MODEL,
      dry: false,
    })
    expect(args).toEqual([
      '--examples',
      '/w/ex.jsonl',
      '--file',
      '/w/notes.txt',
      '--target-model',
      MODEL,
    ])
  })

  test('--dry is added for a plan and left out of the real run', () => {
    expect(typed('--dry Do it.', true)).toEqual(['--file', FILE, '--target-model', MODEL, '--dry'])
    expect(typed('--dry Do it.', false)).toEqual(['--file', FILE, '--target-model', MODEL])
    expect(typed('Do it.', true).at(-1)).toBe('--dry')
  })

  test('no part of a hostile prompt reaches the command line', () => {
    const argv = cliArgv(ROOT, typed(HOSTILE))
    for (const marker of ['PWNED', 'whoami', '$(', '"hi"', 'Ignore', '\n']) {
      expect(argv.some((part) => part.includes(marker)), marker).toBe(false)
    }
    expect(argv.filter((part) => part === '--dry' || part === '--target-model')).toEqual([
      '--target-model',
    ])
    expect(argv).toContain(FILE)
  })

  test('resume and clean take only the run id', () => {
    expect(resumeArgs('20261004-132507-1a2b3c4d')).toEqual(['--resume', '20261004-132507-1a2b3c4d'])
    expect(cleanArgs('20261004-132507-1a2b3c4d')).toEqual(['clean', '20261004-132507-1a2b3c4d'])
  })
})

describe('the dry run first', () => {
  test('above 2,000 characters, not at 2,000', () => {
    expect(DRY_FIRST_CHARS).toBe(2000)
    expect(needsDryFirst(run('x'.repeat(2000)), null)).toBe(false)
    expect(needsDryFirst(run('x'.repeat(2001)), null)).toBe(true)
  })

  test('counts characters, not UTF-16 units', () => {
    expect(needsDryFirst(run('\u{1F600}'.repeat(2000)), null)).toBe(false)
    expect(needsDryFirst(run('\u{1F600}'.repeat(2001)), null)).toBe(true)
  })

  test('never when --dry is asked, and for a --file by its size in bytes', () => {
    expect(needsDryFirst(run(`--dry ${'x'.repeat(5000)}`), null)).toBe(false)
    expect(needsDryFirst(run('--file a.txt'), 2001)).toBe(true)
    expect(needsDryFirst(run('--file a.txt'), 2000)).toBe(false)
    expect(needsDryFirst(run('--file a.txt'), null)).toBe(false)
  })
})

describe('the environment', () => {
  test('sets only UV_PROJECT_ENVIRONMENT', () => {
    expect(childEnv('/c/autoimprover/venv')).toEqual({
      UV_PROJECT_ENVIRONMENT: '/c/autoimprover/venv',
    })
  })

  test('the venv is in the cache folder, never inside the plugin folder', () => {
    const venv = venvFolder({ home: '/h/u', cacheHome: undefined })
    expect(venv).toBe('/h/u/.cache/autoimprover/venv')
    expect(venv?.startsWith(ROOT)).toBe(false)
    expect(venvFolder({ home: '/h/u', cacheHome: '/c/' })).toBe('/c/autoimprover/venv')
  })

  test('a relative XDG_CACHE_HOME is ignored; no absolute folder at all is null', () => {
    expect(venvFolder({ home: '/h/u', cacheHome: 'rel' })).toBe('/h/u/.cache/autoimprover/venv')
    expect(venvFolder({ home: undefined, cacheHome: undefined })).toBeNull()
    expect(venvFolder({ home: 'rel', cacheHome: '' })).toBeNull()
  })

  test('the prompt folder is made under an absolute TMPDIR, else /tmp', () => {
    expect(tmpTemplate('/t/x/')).toBe('/t/x/autoimprover-prompt.XXXXXX')
    expect(tmpTemplate(undefined)).toBe('/tmp/autoimprover-prompt.XXXXXX')
    expect(tmpTemplate('rel')).toBe('/tmp/autoimprover-prompt.XXXXXX')
  })
})

describe('paths and the target model', () => {
  test('a path is absolute, under the home for ~/, else under the session folder', () => {
    expect(resolvePath('/a/b.txt', { cwd: '/w', home: '/h/u' })).toBe('/a/b.txt')
    expect(resolvePath('~/b.txt', { cwd: '/w', home: '/h/u' })).toBe('/h/u/b.txt')
    expect(resolvePath('d/b.txt', { cwd: '/w/', home: '/h/u' })).toBe('/w/d/b.txt')
  })

  test('the session model is passed when it is a model id or alias', () => {
    for (const model of [MODEL, 'opus', 'claude-opus-5-5[1m]', ' sonnet ']) {
      expect(targetModelOf(model), model).toBe(model.trim())
    }
    for (const model of ['Opus 5.5', '', '--help', 'a b']) {
      expect(targetModelOf(model), model).toBeNull()
    }
  })
})
