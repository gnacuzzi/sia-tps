import fs from "node:fs/promises";
import path from "node:path";
import { pathToFileURL } from "node:url";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const workspaceDir = "/Users/giannanacuzzi/itba/sia/sia-tps/tp4";
const skillDir = "/Users/giannanacuzzi/.codex/plugins/cache/openai-primary-runtime/presentations/26.1025.11240/skills/presentations";
const buildDir = path.join(workspaceDir, ".codex-build", "campaign-slides-model-guided");
const outputDir = path.join(workspaceDir, "docs");
const finalPath = path.join(outputDir, "diapositivas_27_28_29_modelos_v2.pptx");
const sourceTemplate = path.join(outputDir, "diapositivas_27_28_28bis.pptx");

const { applyPresentationChartFont, finalizePresentation } = await import(
  pathToFileURL(path.join(skillDir, "container_tools", "artifact_tool_utils.mjs")).href
);

const C = {
  navy: "#102B45",
  blue: "#1F6397",
  teal: "#07989B",
  orange: "#ED7D14",
  text: "#202A36",
  gray: "#596A79",
  line: "#D4E0EA",
  soft: "#EEF4F8",
  softer: "#F7F9FB",
  white: "#FFFFFF",
};
const FONT = "Helvetica Neue";
const slideSize = { width: 1280, height: 720 };

function addText(slide, text, position, options = {}) {
  const shape = slide.shapes.add({
    geometry: "textbox",
    position,
    fill: "none",
    line: { style: "solid", fill: "none", width: 0 },
  });
  shape.text = text;
  shape.text.style = {
    typeface: FONT,
    fontSize: options.fontSize ?? 18,
    bold: options.bold ?? false,
    color: options.color ?? C.text,
    alignment: options.alignment ?? "left",
  };
  return shape;
}

function addRect(slide, position, fill, line = "none", radius = false) {
  return slide.shapes.add({
    geometry: radius ? "roundRect" : "rect",
    position,
    fill,
    line: { style: "solid", fill: line, width: line === "none" ? 0 : 1 },
    ...(radius ? { borderRadius: "rounded-lg" } : {}),
  });
}

function addBase(slide, title, subtitle, pageLabel) {
  slide.background.fill = C.white;
  addRect(slide, { left: 0, top: 0, width: 1280, height: 11 }, C.orange);
  addText(slide, title, { left: 78, top: 42, width: 1120, height: 52 }, {
    fontSize: 38,
    bold: true,
    color: C.navy,
  });
  if (subtitle) {
    addText(slide, subtitle, { left: 82, top: 109, width: 1110, height: 30 }, {
      fontSize: 17,
      color: C.gray,
      alignment: "center",
    });
  }
  addText(slide, "Ejercicio 1 · Campañas", { left: 78, top: 686, width: 240, height: 18 }, {
    fontSize: 12,
    color: C.gray,
  });
  addText(slide, pageLabel, { left: 1160, top: 686, width: 40, height: 18 }, {
    fontSize: 12,
    color: C.gray,
    alignment: "right",
  });
}

function addStage(slide, x, number, label, headline, body, color) {
  addText(slide, `${number}`, { left: x, top: 170, width: 32, height: 30 }, {
    fontSize: 19,
    bold: true,
    color,
    alignment: "center",
  });
  addText(slide, label.toUpperCase(), { left: x + 38, top: 169, width: 168, height: 25 }, {
    fontSize: 15,
    bold: true,
    color,
  });
  addText(slide, headline, { left: x, top: 228, width: 210, height: 64 }, {
    fontSize: 22,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  addText(slide, body, { left: x, top: 310, width: 210, height: 100 }, {
    fontSize: 15,
    color: C.gray,
    alignment: "center",
  });
}

function addSlide1(presentation) {
  const slide = presentation.slides.add();
  addBase(
    slide,
    "Los modelos delimitan el perfil antes de comparar métricas",
    "La selección sigue una secuencia reproducible y evita elegir sobre todo el catálogo",
    "27",
  );

  addRect(slide, { left: 83, top: 198, width: 1110, height: 4 }, C.line);
  const xs = [85, 315, 545, 775, 1005];
  const colors = [C.orange, C.teal, C.blue, C.orange, C.navy];
  for (let i = 0; i < xs.length; i += 1) {
    addRect(slide, { left: xs[i] + 91, top: 190, width: 18, height: 18 }, colors[i], "none", true);
  }

  addStage(slide, xs[0], "1", "Objetivo", "Define el perfil buscado", "Ventas prioriza escala y descubrimiento. Perros prioriza profundidad y acción.", colors[0]);
  addStage(slide, xs[1], "2", "Kohonen", "Delimita la región temática", "Sólo entran perfiles de la región correspondiente con asignación estable o fronteriza.", colors[1]);
  addStage(slide, xs[2], "3", "Afinidad", "Conserva la mitad más temática", "El corte es la mediana de cada región. Afinidad es el porcentaje de contenido sobre el tema.", colors[2]);
  addStage(slide, xs[3], "4", "PCA y Oja", "Elige escala o profundidad", "PC1 alto para ventas. PC1 bajo para perros. Oja confirma el mismo eje.", colors[3]);
  addStage(slide, xs[4], "5", "Métrica", "Desempata dentro del perfil", "Alcance nuevo relevante para ventas. Visitas nuevas relevantes para perros.", colors[4]);

  addRect(slide, { left: 83, top: 456, width: 1110, height: 1 }, C.line);
  addText(slide, "Qué aporta el análisis", { left: 82, top: 482, width: 250, height: 30 }, {
    fontSize: 19,
    bold: true,
    color: C.navy,
  });
  addText(slide, "Los modelos determinan quién puede competir. Las métricas originales eligen entre esos perfiles.", { left: 330, top: 480, width: 840, height: 32 }, {
    fontSize: 20,
    bold: true,
    color: C.navy,
  });
  addText(slide, "No se interpreta Oja como un segundo voto ni Kohonen como un ranking automático.", { left: 330, top: 527, width: 840, height: 26 }, {
    fontSize: 16,
    color: C.gray,
  });

  slide.speakerNotes.text = [
    "Significado: la decisión no comienza con una fórmula comercial aplicada a todo el catálogo.",
    "Aporte: Kohonen restringe el tipo de perfil, PCA/Oja define la dirección y las métricas originales resuelven dentro de ese conjunto.",
    "Fuentes: docs/pca-creadores/README.md; docs/oja-creadores/README.md; docs/kohonen-experimentos/07-interpretation/README.md; docs/campanas-seleccion-guiada-modelos/README.md.",
  ].join("\n");
  slide.speakerNotes.setVisible(true);
}

function styleTable(table) {
  table.borders.assign({ style: "solid", fill: C.line, width: 1 });
  table.cells.block({ row: 0, column: 0, rowCount: 1, columnCount: 5 }).assign({
    fill: C.navy,
    textStyle: { typeface: FONT, fontSize: 15, bold: true, color: C.white },
    anchor: "middle",
  });
  table.cells.block({ row: 1, column: 0, rowCount: 4, columnCount: 5 }).assign({
    textStyle: { typeface: FONT, fontSize: 14, color: C.text },
    anchor: "middle",
  });
  for (const row of [1, 3]) {
    table.cells.block({ row, column: 0, rowCount: 1, columnCount: 5 }).fill = C.soft;
  }
  for (let row = 1; row <= 4; row += 1) {
    table.getCell(row, 0).text.style = { typeface: FONT, fontSize: 15, bold: true, color: C.text, alignment: "center" };
    table.getCell(row, 2).text.style = { typeface: FONT, fontSize: 15, bold: true, color: C.blue, alignment: "center" };
  }
  table.getCell(4, 1).text.style = { typeface: FONT, fontSize: 14, bold: true, color: C.orange, alignment: "center" };
}

function addSlide2(presentation) {
  const slide = presentation.slides.add();
  addBase(
    slide,
    "Los modelos cambian tres de las cuatro selecciones",
    "La métrica ligada al objetivo se aplica sólo después de verificar región, afinidad y PC1",
    "28",
  );

  const values = [
    ["Campaña", "Perfil exigido", "Seleccionado", "Evidencia de los modelos", "Resultado estimado"],
    ["Fútbol", "Escala", "@golconmate", "Estable 100 %\nAfinidad 82,2 % · PC1 +3,34", "102 mil cuentas\nnuevas y afines"],
    ["Belleza", "Escala", "@glamencasa", "Estable 100 %\nAfinidad 74,1 % · PC1 +1,91", "25,7 mil cuentas\nnuevas y afines"],
    ["Perros", "Profundidad", "@guiadehuellas", "Estable 100 %\nAfinidad 68,8 % · PC1 −2,41", "51 visitas nuevas\ny afines"],
    ["Arte", "Escala", "@pincelcotidiano", "Fronterizo 70 %\nAfinidad 70,2 % · PC1 +0,69", "23,5 mil cuentas\nnuevas y afines"],
  ];
  const table = slide.tables.add({
    rows: 5,
    columns: 5,
    left: 70,
    top: 165,
    width: 1140,
    height: 352,
    columnWidths: [122, 150, 190, 350, 328],
    values,
  });
  styleTable(table);

  addText(slide, "Qué significa el resultado", { left: 84, top: 550, width: 250, height: 25 }, {
    fontSize: 17,
    bold: true,
    color: C.blue,
  });
  addText(slide, "El ganador maximiza la señal del objetivo dentro del perfil definido por los modelos. No es el máximo global del catálogo.", { left: 84, top: 584, width: 830, height: 44 }, {
    fontSize: 16,
    color: C.gray,
  });
  addRect(slide, { left: 925, top: 548, width: 278, height: 82 }, "#FFF3E8", C.orange, true);
  addText(slide, "Afinidad temática", { left: 944, top: 562, width: 240, height: 22 }, {
    fontSize: 15,
    bold: true,
    color: C.orange,
    alignment: "center",
  });
  addText(slide, "porcentaje del contenido dedicado al tema", { left: 944, top: 590, width: 240, height: 25 }, {
    fontSize: 13,
    color: C.gray,
    alignment: "center",
  });

  slide.speakerNotes.text = [
    "Significado: cada fila muestra el creador que gana después de pasar los filtros de región, afinidad y dirección de PC1.",
    "Aporte: la tabla permite justificar la elección con resultados de Kohonen y PCA/Oja, no sólo con métricas originales.",
    "Los resultados son proxies descriptivos. No predicen ventas ni adopciones.",
    "Fuente: docs/campanas-seleccion-guiada-modelos/generated/seleccion_final.csv.",
  ].join("\n");
  slide.speakerNotes.setVisible(true);
}

function addComparisonChart(slide, config) {
  const chart = slide.charts.add("bar", {
    position: config.position,
    categories: ["Arte", "Perros", "Belleza", "Fútbol"],
    series: [
      { name: "@mateconideas", values: config.baseline, fill: C.orange },
      { name: "Selección por modelos", values: config.selected, fill: C.blue },
    ],
    barOptions: { direction: "bar", grouping: "clustered", gapWidth: 55 },
    hasLegend: true,
    legend: { position: "bottom", overlay: false, textStyle: { fontSize: 11, fill: C.gray } },
    xAxis: {
      min: 0,
      max: config.max,
      majorUnit: config.major,
      numberFormatCode: config.format,
      textStyle: { fontSize: 11, fill: C.gray },
      majorGridlines: { style: "solid", fill: C.line, width: 1 },
      line: { style: "solid", fill: C.line, width: 1 },
    },
    yAxis: {
      textStyle: { fontSize: 11, fill: C.gray },
      line: { style: "solid", fill: C.line, width: 1 },
    },
    dataLabels: {
      showValue: true,
      position: "outEnd",
      numberFormatCode: config.format,
      textStyle: { fontSize: 10, fill: C.text, bold: true },
    },
    chartFill: C.white,
    chartLine: { style: "solid", fill: "none", width: 0 },
    plotAreaFill: C.white,
    plotAreaLine: { style: "solid", fill: "none", width: 0 },
  });
  applyPresentationChartFont(chart, { fontFamily: FONT });
  addText(slide, config.title, { left: config.position.left, top: config.position.top - 34, width: config.position.width, height: 28 }, {
    fontSize: 19,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
}

function addSlide3(presentation) {
  const slide = presentation.slides.add();
  addBase(
    slide,
    "Más afinidad y menor precio, con distinto volumen",
    "Cada seleccionado se compara con @mateconideas, el creador con más followers del catálogo",
    "29",
  );

  addComparisonChart(slide, {
    title: "Afinidad temática (%)",
    position: { left: 72, top: 190, width: 555, height: 280 },
    baseline: [15.3, 13.23, 61.58, 9.89],
    selected: [70.19, 68.83, 74.13, 82.18],
    max: 90,
    major: 10,
    format: "0.0",
  });
  addComparisonChart(slide, {
    title: "Precio por publicación (miles de USD)",
    position: { left: 675, top: 190, width: 535, height: 280 },
    baseline: [6.31069, 6.31069, 6.31069, 6.31069],
    selected: [2.36186, 0.18869, 2.92231, 3.2512],
    max: 7,
    major: 1,
    format: "0.0",
  });

  addRect(slide, { left: 70, top: 515, width: 1140, height: 1 }, C.line);
  addText(slide, "Volumen estimado ligado al objetivo · selección / @mateconideas", { left: 200, top: 532, width: 880, height: 25 }, {
    fontSize: 17,
    bold: true,
    color: C.navy,
    alignment: "center",
  });
  const blocks = [
    ["FÚTBOL", "102 mil / 11 mil", "cuentas nuevas y afines", C.blue],
    ["BELLEZA", "25,7 mil / 70,4 mil", "cuentas nuevas y afines", C.blue],
    ["PERROS", "51 / 248", "visitas nuevas y afines", C.teal],
    ["ARTE", "23,5 mil / 17,5 mil", "cuentas nuevas y afines", C.blue],
  ];
  blocks.forEach((block, index) => {
    const x = 75 + index * 285;
    if (index > 0) addRect(slide, { left: x - 10, top: 573, width: 1, height: 70 }, C.line);
    addText(slide, block[0], { left: x, top: 572, width: 265, height: 20 }, { fontSize: 14, bold: true, color: block[3], alignment: "center" });
    addText(slide, block[1], { left: x, top: 602, width: 265, height: 22 }, { fontSize: 17, bold: true, color: C.text, alignment: "center" });
    addText(slide, block[2], { left: x, top: 627, width: 265, height: 20 }, { fontSize: 13, color: C.gray, alignment: "center" });
  });
  addText(slide, "Belleza y perros resignan volumen estimado para conservar el perfil aprendido por los modelos y reducir el costo.", { left: 170, top: 657, width: 940, height: 20 }, {
    fontSize: 13,
    bold: true,
    color: C.navy,
    alignment: "center",
  });

  slide.speakerNotes.text = [
    "Significado: la selección por modelos aumenta la afinidad temática y reduce el precio en las cuatro campañas.",
    "Aporte: la comparación muestra el costo de esa decisión. En belleza y perros, el creador con más followers conserva mayor volumen estimado; la propuesta prioriza coherencia de perfil y costo.",
    "Las cifras son proxies descriptivos, no conversiones observadas.",
    "Fuentes: datos/creadores.csv; docs/campanas-seleccion-guiada-modelos/generated/seleccion_final.csv.",
  ].join("\n");
  slide.speakerNotes.setVisible(true);
}

async function writeBlob(filePath, blob) {
  await fs.writeFile(filePath, new Uint8Array(await blob.arrayBuffer()));
}

async function main() {
  await fs.mkdir(buildDir, { recursive: true });
  await fs.mkdir(outputDir, { recursive: true });

  const presentation = Presentation.create({ slideSize });
  addSlide1(presentation);
  addSlide2(presentation);
  addSlide3(presentation);

  for (const [index, slide] of presentation.slides.items.entries()) {
    const preview = await presentation.export({ slide, format: "png", scale: 1.5 });
    await writeBlob(path.join(buildDir, `slide-${index + 1}.png`), preview);
  }
  const montage = await presentation.export({ format: "webp", montage: true, scale: 1 });
  await writeBlob(path.join(buildDir, "montage.webp"), montage);

  const candidatePath = path.join(buildDir, "candidate.pptx");
  await (await PresentationFile.exportPptx(presentation)).save(candidatePath);
  await finalizePresentation({
    explicitTotalSlideCount: 3,
    requiredNativeTableOwnerSlides: [2],
    requiredNativeChartOwnerSlides: [3],
    materializeLiteralChartWorkbooks: true,
    nativeChartTargetApplication: "powerpoint",
    workspaceDir,
    candidatePath,
    finalPath,
    pythonExecutable: "/Users/giannanacuzzi/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3",
    integrityValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_package_integrity.py"),
    layoutValidatorPath: path.join(skillDir, "container_tools", "inspect_presentation_layout_geometry.py"),
    layoutArgs: [
      "--expected-slide-size-emu", "12192000,6858000",
      "--validate-heading-fit",
      "--require-native-table-slide", "2",
    ],
    fontPolicy: {
      basis: "reference",
      families: [FONT],
      referencePath: sourceTemplate,
      referenceSha256: "3b9fe82e3a4d7336024219bf17ab2f81cc543f2ccecddf483084b1d1b422dcd8",
    },
    verifyArtifactToolImport: true,
    receiptPath: path.join(buildDir, "validation-v2.json"),
  });
  console.log(finalPath);
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
