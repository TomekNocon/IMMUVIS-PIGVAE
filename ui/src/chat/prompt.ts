// What the language model is told, and the (image-free) model output it may use.
import type { PredictResponse } from '@/api'

export const SYSTEM_PROMPT = `You are an assistant inside a research web app. The app runs a machine-learning model that predicts the lung cancer subtype (Adenocarcinoma or Squamous cell carcinoma) from one imaging mass cytometry (IMC) image of tumour tissue. The image is cut into 128 x 128 px crops; each crop is embedded (ImmuVis encoder, then PCA, then the PIGVAE graph autoencoder) and a gated attention MIL head (ABMIL) combines the crops into one prediction.

You get the model's output for one image as JSON in <model_output>:
- probabilities and logit: the prediction and how sure the model is.
- crops (row, col = top-left pixel of the crop): attention is the share of the model's weighting (sums to 1 over all crops); ig is Integrated Gradients, the crop's signed contribution to the logit: negative pushes toward Adenocarcinoma, positive toward Squamous cell carcinoma. Rank 1 = largest.
- model: the cross-validation fold of the prediction head and its validation AUROC.

Rules:
1. Use only this data plus general knowledge of how these methods work. You cannot see the image, the cells or marker intensities; say so when a question needs them.
2. This is a research tool, not a diagnosis. Never give medical advice or clinical recommendations; if asked, say the output must not be used for patient care.
3. Do not invent biology. If you mention what usually distinguishes the subtypes, label it as general background, not something this output shows.
4. Be concise: plain language, at most about 150 words unless the user asks for more; short paragraphs or "-" bullet points, **bold** sparingly, no tables or headings. Refer to crops as "row R, col C". Quote numbers as in the data (percentages for probabilities and attention).
5. Flag uncertainty: probabilities between 30% and 70%, or attention and IG that disagree, deserve a caveat. High attention does not mean a crop pushed the decision; IG says which way it pushed.`

const r = (x: number, digits = 4) => Number(x.toPrecision(digits))

/** The prediction as compact JSON for the model: numbers only, no preview image. */
export function buildContext(res: PredictResponse) {
  const classes = Object.keys(res.probabilities)
  const byAtt = [...res.crops].sort((a, b) => b.attention - a.attention)
  const byIg = [...res.crops].sort((a, b) => Math.abs(b.ig_importance ?? 0) - Math.abs(a.ig_importance ?? 0))
  const igRank = new Map(byIg.map((c, i) => [c, i + 1]))
  const toward = (ig: number) => (ig > 0 ? classes[1] : classes[0])
  return {
    task: `${res.label}: lung cancer subtype from one IMC image (cords panel)`,
    classes,
    prediction: res.prediction,
    probabilities: Object.fromEntries(classes.map((c) => [c, r(res.probabilities[c])])),
    logit: Array.isArray(res.logit) ? res.logit.map((x) => r(x)) : r(res.logit),
    ig_sign: `ig < 0 pushes toward ${classes[0]}, ig > 0 toward ${classes[1]}`,
    image: {
      width: res.image.width,
      height: res.image.height,
      tile_size: res.image.tile_size,
      n_crops: res.image.n_crops,
      note: 'strips at the right/bottom not covered by whole crops were not analysed',
    },
    model: res.model,
    summary: {
      crops_toward: Object.fromEntries(
        classes.map((c) => [c, res.crops.filter((x) => toward(x.ig_importance ?? 0) === c).length]),
      ),
      top_crop_attention_share: r(byAtt[0]?.attention ?? 0, 3),
    },
    crops: byAtt.map((c, i) => ({
      row: c.row,
      col: c.col,
      attention: r(c.attention, 3),
      attention_rank: i + 1,
      ig: c.ig_importance == null ? null : r(c.ig_importance, 3),
      ig_rank: igRank.get(c),
    })),
  }
}

export const STARTERS = [
  'Why did the model predict this?',
  'Explain the most important crop.',
  'How confident is this prediction?',
  'What is the difference between attention and IG here?',
]
