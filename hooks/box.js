// What the end of a run does with the prompt box (SPEC R21, amended 2026-10-05): an improved
// prompt goes in by itself as an editable draft when the box is empty; a draft the person typed
// is never overwritten (the pane's Use it replaces it on request); a result that is not
// holdout-verified says so. Nothing is sent: the person presses Enter. Pure: the caller reads
// and fills the box, and shows the words chosen here in the pane, the text and the toast.

/** @import { View } from './report.js' */
/** @typedef {{ fill: boolean, box: string, toast: string }} BoxPlan */

/**
 * Whether to fill the box with the improved prompt, and what to say; null when the run has no
 * improved prompt (a kept original, a failure, a cancel, a plan).
 * @param {View} view how the run ended
 * @param {string} draft what the prompt box holds now
 * @returns {BoxPlan | null}
 */
export function boxPlan(view, draft) {
  if (view.improved === null) return null
  if (draft.trim() === '') {
    return said(true, `the ${what(view)} is in the prompt box: edit it and press Enter to send`)
  }
  return said(false, `your draft was kept; press Use it to replace it with the ${what(view)}`)
}

/**
 * What to say when the box did not take the prompt (a dialog held the keys, or there is no box).
 * @param {View} view
 * @param {string | undefined} refusal the engine's reason, when it gave one
 * @returns {BoxPlan}
 */
export function boxRefused(view, refusal) {
  const why = refusal ?? 'refused'
  return said(false, `the prompt box did not take the ${what(view)} (${why}); press Use it or Copy`)
}

/** @param {View} view */
function what(view) {
  return view.verified ? 'improved prompt' : 'improved prompt (NOT verified)'
}

/** @param {boolean} fill @param {string} text @returns {BoxPlan} */
function said(fill, text) {
  return { fill, box: text, toast: `autoimprover: ${text}` }
}
