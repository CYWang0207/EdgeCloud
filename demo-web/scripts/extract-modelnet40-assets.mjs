import { createRequire } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const sharp = require("/Users/wangchengyang/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/sharp");

const sourceRoot = "/Users/wangchengyang/Documents/EdgeCloud/modelnet40_demo_4view_selected";
const scriptDir = path.dirname(fileURLToPath(import.meta.url));
const outputRoot = path.resolve(scriptDir, "../assets/modelnet40");

const cases = [
  {
    id: "night_stand_0224",
    cleanDir: "recovered/night_stand_0224",
    cleanPattern: "view_0{view}_night_stand_0224_00{view}.png",
    composite: "drift_visualization_night_stand_0224.png",
  },
  {
    id: "bathtub_0134",
    cleanDir: "recovered/bathtub_0134",
    cleanPattern: "view_0{view}_bathtub_0134_00{view}.png",
    composite: "drift_visualization_bathtub_0134.png",
  },
];

const cropColumns = [42, 435, 827, 1219];
const cropRows = {
  illumination: 489,
  defocus: 858,
  sensor_noise: 1227,
};
const cropSize = 338;

for (const sample of cases) {
  for (let view = 1; view <= 4; view += 1) {
    const cleanName = sample.cleanPattern.replaceAll("{view}", String(view));
    await sharp(path.join(sourceRoot, sample.cleanDir, cleanName))
      .webp({ quality: 88 })
      .toFile(path.join(outputRoot, sample.id, "clean", `view-${view}.webp`));
  }

  for (const [condition, top] of Object.entries(cropRows)) {
    for (let view = 1; view <= 4; view += 1) {
      await sharp(path.join(sourceRoot, sample.composite))
        .extract({ left: cropColumns[view - 1], top, width: cropSize, height: cropSize })
        .resize(224, 224)
        .webp({ quality: 88 })
        .toFile(path.join(outputRoot, sample.id, condition, `view-${view}.webp`));
    }
  }
}

console.log(`Exported ModelNet40 web assets to ${outputRoot}`);
