// The report pane's tree (SPEC R21): the head rows of a view (report.js: the result, the prompt
// box, whether it is verified, the improved prompt in full and wrapped), then the buttons the
// caller made, then the details; and how many rows the pane asks for, so the improved prompt is
// seen whole. Pure: the elements come from the table the caller resolved for the surface, and
// `h` is the environment's element factory.

import { rowsOf } from './report.js'

/** @import { RenderElement } from 'claude-code' */
/** @import { Row, View } from './report.js' */

/** How each kind of row is drawn. @type {Record<Row['kind'], Record<string, unknown>>} */
const STYLES = {
  title: { bold: true },
  heading: { bold: true },
  hint: { color: 'yellow' },
  dim: { dimColor: true },
  line: {},
  code: {},
}
// The width the row count assumes for wrapped text, and the most rows the pane asks for.
const COLUMNS = 76
const MOST_ROWS = 120

/**
 * The pane for `view` (a placeholder before the first run).
 * @param {View | null} view
 * @param {{ Box: any, Text: any, Code: any }} elements the surface's, as the render hook resolved
 * @param {unknown[]} buttons
 * @returns {RenderElement}
 */
export function paneTree(view, elements, buttons) {
  const { Box, Text, Code } = elements
  if (view === null) {
    return /** @type {RenderElement} */ (h(Text, { dimColor: true }, 'No autoimprover run yet.'))
  }
  const { head, tail } = rowsOf(view, false)
  /** @param {Row} row */
  const draw = (row) =>
    row.kind === 'code'
      ? h(Code, { source: row.text, wrap: 'wrap' })
      : h(Text, { ...STYLES[row.kind], wrap: 'wrap' }, row.text)
  const bar = h(Box, { flexDirection: 'row', gap: 2, marginY: 1 }, ...buttons)
  const column = { flexDirection: 'column' }
  return /** @type {RenderElement} */ (h(Box, column, ...head.map(draw), bar, ...tail.map(draw)))
}

/**
 * The rows the pane wants: every line of the view wrapped at COLUMNS, and the buttons.
 * @param {View} view
 * @returns {number}
 */
export function paneRows(view) {
  const { head, tail } = rowsOf(view, false)
  let rows = 3
  for (const row of [...head, ...tail]) {
    for (const line of row.text.split('\n')) rows += Math.max(1, Math.ceil(line.length / COLUMNS))
  }
  return Math.min(rows, MOST_ROWS)
}
