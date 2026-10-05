// The child's output as it streams in (SPEC R2, R21): stdout is collected whole (the one JSON
// object), stderr is cut into lines for the status line, and the `run folder: <path>` line the
// CLI prints is caught the first time it appears, so the run id is known (and kept for
// `--resume`) while the run still goes. Pure: the caller feeds it the chunks.

import { cleanText } from './report.js'

// The bytes of stderr kept for the report of a run that printed no JSON (the tail matters).
const STDERR_KEPT = 65536
const RUN_FOLDER = /^run folder: (\/.*?)(?: \(remove it with: autoimprover clean \S+\))?$/

export class Progress {
  constructor() {
    this.stdout = ''
    this.stderr = ''
    /** @type {string | null} */
    this.runDir = null
    this.partial = ''
  }

  /**
   * Takes one chunk; says what it changed: the status line (the last non-blank stderr line in
   * it) and the run folder when this chunk revealed it.
   * @param {{ stream: 'stdout' | 'stderr', text: string }} chunk
   * @returns {{ status: string | null, runDir: string | null }}
   */
  push(chunk) {
    if (chunk.stream === 'stdout') {
      this.stdout += chunk.text
      return { status: null, runDir: null }
    }
    this.stderr = (this.stderr + chunk.text).slice(-STDERR_KEPT)
    const lines = (this.partial + chunk.text).split('\n')
    this.partial = lines.pop() ?? ''
    return this.read(lines)
  }

  /**
   * The last line, when the child ended without a newline after it.
   * @returns {{ status: string | null, runDir: string | null }}
   */
  finish() {
    const lines = [this.partial]
    this.partial = ''
    return this.read(lines)
  }

  /**
   * @param {string[]} lines
   * @returns {{ status: string | null, runDir: string | null }}
   */
  read(lines) {
    let status = null
    let runDir = null
    for (const line of lines) {
      const said = cleanText(line).trim()
      if (said === '') continue
      status = said.length > 160 ? `${said.slice(0, 157)}...` : said
      const found = RUN_FOLDER.exec(said)?.[1] ?? null
      if (found !== null && this.runDir === null) {
        this.runDir = found
        runDir = found
      }
    }
    return { status, runDir }
  }
}
