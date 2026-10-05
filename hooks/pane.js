// The report pane's tree (SPEC R21): the rows of a view (report.js) as the surface's elements,
// then the buttons the caller made. Pure: the elements come from the table the caller resolved
// for the surface, and `h` is the environment's element factory.

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

/**
 * The pane for `view` (a placeholder before the first run): its rows, then `buttons` in a row.
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
  const rows = rowsOf(view, false).map((row) =>
    row.kind === 'code'
      ? h(Code, { source: row.text, wrap: 'wrap' })
      : h(Text, STYLES[row.kind], row.text)
  )
  const bar = h(Box, { flexDirection: 'row', gap: 2, marginTop: 1 }, ...buttons)
  return /** @type {RenderElement} */ (h(Box, { flexDirection: 'column' }, ...rows, bar))
}
