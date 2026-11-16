// Presentación del TP5 con la plantilla de los TP3 y TP4 del grupo 10.
// Uso (desde tp5/): node presentacion/build_deck.js
const path = require("path");
const pptxgen = require("pptxgenjs");

const DOCS = path.join(__dirname, "..", "docs");
const FIG = (name) => path.join(DOCS, name);
const OUT = path.join(DOCS, "TP5_SIA_GRUPO_10.pptx");

// Paleta y tipografía de las presentaciones anteriores.
const C = {
  navy: "102A43", orange: "E67E22", teal: "159A9C", blue: "1F5D8F",
  green: "23856D", red: "C0392B", text: "1F2933", gray: "52616B",
  line: "D9E2EC", soft: "F1F5F8", softOrange: "FDF1E6", white: "FFFFFF",
  muted: "B8C4CE",
};
// QA_FONT permite renderizar la vista previa con una fuente métricamente cercana.
const FONT = process.env.QA_FONT || "Helvetica Neue";
const MONO = "Courier New";
const W = 13.333, H = 7.5;
const X0 = 0.7, XW = W - 2 * X0; // margen y ancho útil
const TOP = 1.65, BOTTOM = 6.85;

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "Grupo 10";
pres.title = "TP5 - Autoencoders";
pres.theme = { headFontFace: FONT, bodyFontFace: FONT };

// --- Layouts -------------------------------------------------------------------
pres.defineSlideMaster({
  title: "PORTADA",
  background: { color: C.navy },
  objects: [{ rect: { x: 0, y: 0, w: W, h: 0.16, fill: { color: C.teal } } }],
});

const SECTIONS = {
  A: { color: C.orange, footer: "Ejercicio 1A · Autoencoder básico" },
  B: { color: C.teal, footer: "Ejercicio 1B · Denoising autoencoder" },
  C: { color: C.blue, footer: "Ejercicio 2 · Autoencoder variacional" },
  Z: { color: C.navy, footer: "Cierre" },
};

for (const [key, section] of Object.entries(SECTIONS)) {
  pres.defineSlideMaster({
    title: `SEPARADOR_${key}`,
    background: { color: C.navy },
    objects: [{ rect: { x: 0, y: 0, w: 0.16, h: H, fill: { color: section.color } } }],
  });
  pres.defineSlideMaster({
    title: `CONTENIDO_${key}`,
    background: { color: C.white },
    objects: [
      { rect: { x: 0, y: 0, w: W, h: 0.1, fill: { color: section.color } } },
      { text: { text: section.footer, options: { x: X0, y: 7.02, w: 6, h: 0.25,
        fontFace: FONT, fontSize: 10, color: C.gray, margin: 0 } } },
      { placeholder: { options: { name: "title", type: "title", x: X0, y: 0.38, w: XW, h: 0.7,
        fontFace: FONT, fontSize: 27, bold: true, color: C.navy, valign: "middle", align: "left", margin: 0 },
        text: "" } },
    ],
    slideNumber: { x: W - X0 - 0.6, y: 7.02, w: 0.6, h: 0.25, fontFace: FONT, fontSize: 10,
      color: C.gray, align: "right" },
  });
}

// --- Ayudas --------------------------------------------------------------------
function text(slide, value, opts) {
  slide.addText(value, { fontFace: FONT, color: C.text, margin: 0, isTextBox: true,
    valign: "top", ...opts });
}

function content(section, title, subtitle, notes) {
  const slide = pres.addSlide({ masterName: `CONTENIDO_${section}` });
  slide.addText(title, { placeholder: "title" });
  if (subtitle) text(slide, subtitle, { x: X0, y: 1.12, w: XW, h: 0.35, fontSize: 13, color: C.gray });
  if (notes) slide.addNotes(notes);
  return slide;
}

function divider(section, number, title, subtitle, notes) {
  const slide = pres.addSlide({ masterName: `SEPARADOR_${section}` });
  const color = SECTIONS[section].color;
  text(slide, number, { x: 1.0, y: 1.6, w: 3, h: 1.0, fontSize: 60, bold: true, color });
  text(slide, title, { x: 1.0, y: 2.75, w: 11, h: 0.9, fontSize: 40, bold: true, color: C.white });
  text(slide, subtitle, { x: 1.0, y: 3.95, w: 10.5, h: 0.9, fontSize: 20, color: "D7E1EA" });
  if (notes) slide.addNotes(notes);
  return slide;
}

// Coloca una imagen dentro de una caja, manteniendo su proporción.
function image(slide, file, box, ratio) {
  let w = box.w, h = w / ratio;
  if (h > box.h) { h = box.h; w = h * ratio; }
  const x = box.x + (box.w - w) / 2;
  const y = box.y + (box.h - h) / 2;
  slide.addImage({ path: FIG(file), x, y, w, h, altText: file });
  return { x, y, w, h };
}

// Columna lateral con bloques etiqueta / titular / detalle, como en el TP4.
function sidebar(slide, items, x = 9.55, y = TOP + 0.1, w = XW + X0 - 9.55) {
  slide.addShape(pres.shapes.LINE, { x: x - 0.3, y: TOP, w: 0, h: BOTTOM - TOP - 0.2,
    line: { color: C.line, width: 2 } });
  const block = (BOTTOM - y - 0.2) / items.length;
  items.forEach((item, i) => {
    const top = y + i * block;
    text(slide, item.label.toUpperCase(), { x, y: top, w, h: 0.3, fontSize: 12, bold: true,
      color: item.color || C.orange });
    text(slide, item.head, { x, y: top + 0.32, w, h: 0.5, fontSize: 19, bold: true, color: C.navy,
      fit: "shrink" });
    if (item.body) text(slide, item.body, { x, y: top + 0.85, w, h: block - 0.95, fontSize: 13,
      color: C.gray });
    if (i < items.length - 1) slide.addShape(pres.shapes.LINE, { x, y: top + block - 0.1, w, h: 0,
      line: { color: C.line, width: 1 } });
  });
}

function takeaway(slide, value, color = C.navy, y = 6.3) {
  text(slide, value, { x: X0, y, w: XW, h: 0.5, fontSize: 16, bold: true, color, align: "center",
    valign: "middle" });
}

function stat(slide, value, label, x, y, w, color = C.blue) {
  text(slide, value, { x, y, w, h: 0.75, fontSize: 36, bold: true, color, align: "center" });
  text(slide, label, { x, y: y + 0.78, w, h: 0.6, fontSize: 13, color: C.gray, align: "center" });
}

// Fila numerada sobre una banda suave, como el "Embudo experimental" del TP3.
function steps(slide, rows, color, y0 = TOP + 0.1, rowH = 0.68) {
  rows.forEach(([head, detail], i) => {
    const y = y0 + i * rowH;
    text(slide, String(i + 1), { x: X0 + 0.1, y: y + 0.05, w: 0.5, h: 0.45, fontSize: 20, bold: true,
      color, align: "center", valign: "middle" });
    slide.addShape(pres.shapes.RECTANGLE, { x: X0 + 0.8, y, w: XW - 0.8, h: rowH - 0.14,
      fill: { color: C.soft } });
    text(slide, head, { x: X0 + 1.05, y, w: 5.2, h: rowH - 0.14, fontSize: 16, bold: true,
      color: C.navy, valign: "middle" });
    text(slide, detail, { x: X0 + 6.3, y, w: XW - 6.7, h: rowH - 0.14, fontSize: 13, color: C.gray,
      valign: "middle", align: "right" });
  });
}

function table(slide, header, rows, opts) {
  const head = header.map((h) => ({ text: h, options: { bold: true, color: C.navy,
    fill: { color: C.white }, border: [{ type: "none" }, { type: "none" },
      { pt: 1.5, color: C.navy }, { type: "none" }] } }));
  const highlight = new Set(opts.highlight || []);
  const body = rows.map((row, r) => row.map((cell) => ({ text: String(cell), options: {
    color: C.text, fill: { color: highlight.has(r) ? C.softOrange : (r % 2 ? C.white : C.soft) },
    bold: highlight.has(r), border: [{ type: "none" }, { type: "none" },
      { pt: 0.75, color: C.line }, { type: "none" }] } })));
  slide.addTable([head, ...body], { x: opts.x ?? X0, y: opts.y ?? TOP + 0.1, w: opts.w ?? XW,
    colW: opts.colW, fontFace: opts.fontFace || FONT, fontSize: opts.fontSize || 14,
    rowH: opts.rowH || 0.42, valign: "middle", margin: [0.04, 0.12, 0.04, 0.12], align: opts.align || "left" });
}

function columns(slide, cols, y = TOP + 0.2, h = 3.6) {
  const gap = 0.4, w = (XW - gap * (cols.length - 1)) / cols.length;
  cols.forEach((col, i) => {
    const x = X0 + i * (w + gap);
    slide.addShape(pres.shapes.LINE, { x, y, w: 0, h, line: { color: col.color, width: 3 } });
    text(slide, col.label, { x: x + 0.25, y, w: w - 0.3, h: 0.35, fontSize: 15, bold: true,
      color: col.color });
    text(slide, col.lines.map((line) => ({ text: line, options: { breakLine: true } })),
      { x: x + 0.25, y: y + 0.5, w: w - 0.3, h: h - 0.5, fontSize: 15, color: C.text,
        paraSpaceAfter: 6 });
  });
}

// ================================================================================
// Portada
{
  const slide = pres.addSlide({ masterName: "PORTADA" });
  text(slide, "Trabajo Práctico N° 5", { x: 1.0, y: 1.25, w: 8, h: 0.5, fontSize: 22, bold: true,
    color: C.teal });
  text(slide, "Autoencoders", { x: 1.0, y: 1.95, w: 11, h: 1.2, fontSize: 54, bold: true,
    color: C.white });
  text(slide, "Autoencoder básico, denoising y variacional sobre letras de font.h y funciones de Python",
    { x: 1.0, y: 3.55, w: 10.5, h: 0.9, fontSize: 21, color: "D7E1EA" });
  text(slide, "Sistemas de Inteligencia Artificial · Grupo 10 · 2Q 2026", { x: 1.0, y: 6.3, w: 8,
    h: 0.35, fontSize: 14, color: "9FB3C8" });
}

// ================================================================================
// 01 · Autoencoder básico
divider("A", "01", "Autoencoder básico",
  "32 caracteres de 7 x 5 píxeles en un espacio latente de dos dimensiones");

{
  const s = content("A", "Objetivo: 32 patrones con a lo sumo un píxel mal cada uno",
    "font.h · cada carácter aplanado a 35 valores binarios · latente de dimensión 2",
    "El criterio es el peor caso: cada uno de los 32 patrones con como máximo un píxel incorrecto. Un promedio menor a 1 podría esconder una letra muy mal reconstruida.");
  image(s, "font-original.png", { x: X0, y: TOP, w: 8.4, h: 5.1 }, 1.644);
  stat(s, "32", "patrones de 7 x 5", 9.55, TOP + 0.15, 3.1, C.navy);
  stat(s, "2", "dimensiones del espacio latente", 9.55, TOP + 1.75, 3.1, C.blue);
  stat(s, "≤ 1", "píxel incorrecto en el peor patrón", 9.55, TOP + 3.35, 3.1, C.orange);
}

{
  const s = content("A", "El embudo experimental del ejercicio 1A",
    "Una novedad por paso · lo prometedor pasa a más semillas · el resto se descarta con evidencia",
    "Arquitectura y optimizador interactúan, por eso no se hizo un producto cartesiano ciego: se calibran tasas, se cruza, se repite con semillas y se refina donde hay tendencia.");
  steps(s, [
    ["Calibración de tasas", "2 redes ancla · 4 optimizadores · 30 corridas"],
    ["Tamiz arquitectura x optimizador", "6 arquitecturas · 4 perfiles · semilla 0"],
    ["Semillas y refinamiento con MSE", "ancho, épocas y profundidad · 3 semillas"],
    ["Cambio de pérdida a BCE", "tasa recalibrada · 10 semillas"],
    ["Arquitecturas con BCE", "9 arquitecturas · 2 tasas · 10 semillas"],
    ["Modelo congelado", "reconstrucción, espacio latente y generación"],
  ], C.orange);
  takeaway(s, "Siempre se evalúa el estado final tras un presupuesto común, nunca el mejor checkpoint",
    C.orange, 6.2);
}

{
  const s = content("A", "Primer intento: más épocas empeoraron el resultado",
    "35-15-2-15-35 · tanh · MSE · Adam 0,01 · 20.000 épocas · semilla 0",
    "Llegó a 3 errores cerca de la época 3.500 y después empeoró hasta 7. De acá sale la regla de evaluar el estado final y no elegir el mejor punto intermedio.");
  s.addChart(pres.charts.LINE, [{ name: "Máximo de píxeles incorrectos",
    labels: ["0", "100", "1.000", "3.493", "10.000", "20.000"], values: [21, 11, 3, 3, 4, 7] }], {
    x: X0, y: TOP, w: 8.5, h: 5.0, chartColors: [C.orange], lineSize: 3, lineDataSymbolSize: 9,
    showValue: true, dataLabelPosition: "t", dataLabelFontSize: 12, dataLabelColor: C.text,
    dataLabelFontFace: FONT, catAxisLabelFontFace: FONT, valAxisLabelFontFace: FONT,
    catAxisLabelColor: C.gray, valAxisLabelColor: C.gray, valGridLine: { color: "E4E7EB", size: 1 },
    catGridLine: { style: "none" }, showLegend: false, showTitle: true,
    title: "Máximo de píxeles incorrectos por época", titleFontFace: FONT, titleFontSize: 13,
    titleColor: C.navy, showCatAxisTitle: true, catAxisTitle: "época", catAxisTitleColor: C.gray,
    catAxisTitleFontSize: 11, valAxisMinVal: 0,
  });
  sidebar(s, [
    { label: "Mínimo", head: "3 errores", body: "Cerca de la época 3.500.", color: C.teal },
    { label: "Final", head: "7 errores", body: "Solo 17 de 32 patrones dentro del límite.", color: C.red },
    { label: "Regla", head: "Estado final", body: "Se compara siempre con un presupuesto común.", color: C.navy },
  ]);
}

{
  const s = content("A", "Cada optimizador necesita su propia tasa",
    "anclas 35-15-2-15-35 y 35-35-2-35-35 · MSE · 2.000 épocas · semilla 0 · α, γ, β₁ = 0,9 y β₂ = 0,999 fijos",
    "Se descartaron tasas lentas o inestables, no optimizadores. La misma tasa fue estable en la red ancha e inestable en la angosta.");
  table(s, ["Optimizador", "Tasas probadas", "Qué se descartó", "Pasa al tamiz"], [
    ["Descenso de gradiente", "0,01 · 0,1 · 0,3 · 0,5 · 1,0", "0,01 lenta · 1,0 inestable en la red de 15", "0,3"],
    ["Momentum", "0,01 · 0,05 · 0,1 · 0,2", "0,2 inestable en la red de 15", "0,01"],
    ["RMSProp", "0,0001 · 0,001 · 0,01", "0,0001 lenta", "0,01"],
    ["Adam", "0,0001 · 0,001 · 0,01", "0,0001 lenta", "0,01"],
  ], { colW: [2.6, 3.3, 4.2, 1.83], rowH: 0.62, fontSize: 15 });
  takeaway(s, "GD 1,0 y Momentum 0,2 divergen en la red de 15 pero mejoran en la de 35: tasa y arquitectura interactúan",
    C.orange, 5.2);
}

{
  const s = content("A", "Tamiz: sin capas ocultas no alcanza y nada llega a 1 error",
    "MSE · 2.000 épocas · semilla 0 · máximo de píxeles incorrectos al final",
    "A0 no tiene capas ocultas y queda en 10-11 errores con cualquier optimizador. A2, A3 y A4 llegan a 2. Adam se desestabiliza en la red profunda.");
  image(s, "ejs-figs/1a-tamiz-arquitectura-optimizador.png", { x: X0, y: TOP - 0.05, w: 8.4, h: 5.2 }, 1.577);
  sidebar(s, [
    { label: "Sin capas ocultas", head: "10-11 errores", body: "A0 falla con los cuatro optimizadores.", color: C.red },
    { label: "Mejores", head: "Máximo 2", body: "A2, A3 y A4, sobre todo con GD 0,3.", color: C.teal },
    { label: "Inestable", head: "Adam en A4", body: "Llega a un buen punto y termina con 6.", color: C.orange },
  ]);
}

{
  const s = content("A", "Con MSE, ninguna configuración cumple en todas las semillas",
    "GD 0,3 con MSE · 2.000-4.000 épocas · cada punto es una semilla",
    "A4 parecía buena con la semilla 0 y terminó con 11 errores en la 2. Con MSE el techo fue cumplir en 2 de 3 semillas, sin importar épocas, ancho o profundidad. El cambio a BCE (naranja) lleva a 10 de 10.");
  image(s, "ejs-figs/1a-embudo-semillas.png", { x: X0, y: TOP - 0.05, w: 8.5, h: 5.2 }, 1.741);
  sidebar(s, [
    { label: "Una semilla engaña", head: "A4: 2 → 11", body: "Parecía buena con la semilla 0.", color: C.red },
    { label: "Techo con MSE", head: "2 de 3 semillas", body: "Ancho 45; 4.000 épocas, 50 o 60 neuronas no lo mejoran.", color: C.orange },
    { label: "Con BCE", head: "10 de 10", body: "Misma red, otra función de pérdida.", color: C.teal },
  ]);
}

{
  const s = content("A", "BCE corrige con fuerza los píxeles saturados y equivocados",
    "35-45-2-45-35 · descenso de gradiente · 2.000 épocas · tasa recalibrada para BCE con semilla 0",
    "Con MSE el gradiente se multiplica por la pendiente de la sigmoide, casi cero si la neurona está saturada. Con BCE esa pendiente se cancela y el gradiente es proporcional al error.");
  text(s, "Píxel que debería valer 1 y la red predice 0,002", { x: X0, y: TOP + 0.05, w: 5.6, h: 0.35,
    fontSize: 15, bold: true, color: C.navy });
  stat(s, "0,005", "gradiente con MSE\n(x' − x) · x'(1 − x')", X0, TOP + 0.6, 2.7, C.red);
  stat(s, "2,0", "gradiente con BCE\nx' − x", X0 + 2.9, TOP + 0.6, 2.7, C.teal);
  text(s, "BCE empuja unas 400 veces más fuerte justo donde la red está segura y equivocada.",
    { x: X0, y: TOP + 2.25, w: 5.6, h: 0.8, fontSize: 14, color: C.gray });
  table(s, ["Tasa (BCE)", "Máximo de errores", "Lectura"], [
    ["0,01", "6", "demasiado lenta"],
    ["0,03", "0", "viable"],
    ["0,1", "0", "viable y más rápida"],
    ["0,3", "22", "inestable"],
  ], { x: 6.9, y: TOP + 0.05, w: 5.73, colW: [1.5, 2.0, 2.23], highlight: [2], rowH: 0.55, fontSize: 15 });
  takeaway(s, "Con BCE y tasa 0,1, la red de ancho 45 reconstruye los 32 patrones sin errores en 10 de 10 semillas",
    C.teal, 5.6);
}

{
  const s = content("A", "Con BCE, solo 45 neuronas cumple en 10 de 10 con una capa",
    "9 arquitecturas · GD con tasas 0,1 y 0,03 · 2.000 épocas · 10 semillas · 180 corridas",
    "Al cambiar la pérdida volvimos a comparar arquitecturas. Con 3 semillas, 20 y 35 neuronas parecían perfectas; con 10 aparecieron sus fallas. D2 empata con la de 45 pero con 50 % más parámetros.");
  image(s, "ejs-figs/1a-bce-arquitecturas.png", { x: X0, y: TOP - 0.05, w: 8.6, h: 5.25 }, 2.054);
  sidebar(s, [
    { label: "Hasta 15 neuronas", head: "0 de 10", body: "Falta capacidad, no épocas.", color: C.red },
    { label: "35 neuronas", head: "9 de 10", body: "Una semilla se desestabiliza al final.", color: C.orange },
    { label: "Elegida: 45", head: "10 de 10", body: "D2 también, pero con 5.197 parámetros contra 3.457.", color: C.teal },
  ]);
}

{
  const s = content("A", "Las curvas por época confirman que 2.000 épocas alcanzan",
    "BCE · mediana y cuartiles de 10 semillas · una capa con GD 0,1 · profundas con GD 0,03",
    "Comparar con un presupuesto común exige ver que cada arquitectura tuvo tiempo. Las que cumplen llegan a 0 antes de la época 1.400 y se quedan ahí; las angostas se estancan desde la mitad.");
  image(s, "ejs-figs/1a-bce-curvas.png", { x: X0, y: TOP - 0.05, w: XW, h: 4.5 }, 2.727);
  takeaway(s, "Las angostas no necesitan más épocas: se estancan muy por encima de 1 error",
    C.navy, 6.25);
}

{
  const s = content("A", "Modelo congelado del ejercicio 1A",
    "Configuración fija para la corrida publicada (semilla 0) y para las 10 semillas de verificación",
    "Elegida por ser la única de una capa oculta que cumple en todas las semillas. El latente es lineal para que z pueda tomar cualquier valor real.");
  columns(s, [
    { label: "Red", color: C.orange, lines: ["35 → 45 → 2 → 45 → 35", "3.457 parámetros",
      "tanh en las ocultas", "lineal en el latente", "logística en la salida"] },
    { label: "Entrenamiento", color: C.teal, lines: ["BCE", "descenso de gradiente, tasa 0,1",
      "full-batch, 2.000 épocas", "sin parada anticipada", "pesos iniciales en [−0,5; 0,5]"] },
  ], TOP + 0.25, 2.6);
  stat(s, "0", "píxeles incorrectos en los 32 patrones, en las 10 semillas", X0, 4.85, XW, C.teal);
}

{
  const s = content("A", "Las 32 reconstrucciones son exactas",
    "Semilla 0 · original arriba y reconstrucción binarizada con umbral 0,5 abajo",
    "La curva oscila durante las primeras 400 épocas porque la tasa 0,1 da pasos grandes; después baja de forma sostenida.");
  image(s, "ae-final-reconstructions.png", { x: X0, y: TOP - 0.1, w: 4.2, h: 5.3 }, 0.755);
  image(s, "ae-final-loss.png", { x: 5.3, y: TOP, w: 7.33, h: 4.1 }, 1.619);
  text(s, "BCE final 0,177 · el requisito se cumple por primera vez entre las épocas 407 y 928 según la semilla",
    { x: 5.3, y: 5.9, w: 7.33, h: 0.6, fontSize: 14, color: C.gray, align: "center" });
}

{
  const s = content("A", "El espacio latente separa los 32 caracteres",
    "encode(x) de cada letra original · semilla 0",
    "Cada punto es el par (z1, z2) que la red asigna a una letra original. Los ejes no significan nada: otra semilla podría rotar o reflejar el mapa. Solo se controlan los 32 puntos, no el espacio entre ellos.");
  image(s, "ae-final-latent.png", { x: X0, y: TOP - 0.15, w: 7.6, h: 5.35 }, 1.145);
  sidebar(s, [
    { label: "Separados", head: "32 códigos distintos", body: "El decoder puede distinguir cada letra.", color: C.teal },
    { label: "Organización", head: "Trazos parecidos", body: "b, h, k, n, p y r quedan juntas a la derecha.", color: C.blue },
    { label: "Sin significado", head: "Ejes arbitrarios", body: "Solo valen las posiciones relativas.", color: C.orange },
    { label: "Huecos", head: "Zonas sin control", body: "El entrenamiento solo fija 32 puntos.", color: C.red },
  ], 8.95, TOP + 0.05, XW + X0 - 8.95);
}

{
  const s = content("A", "Una letra nueva: interpolación entre a y x",
    "z(t) = (1 − t) · z_a + t · z_b · salida continua arriba y binarizada abajo · semilla 0",
    "Primero se probó m-n, pero todos los puntos se binarizaban como m o n. Con a-x, el punto medio genera un patrón que no está entre los 32, a un píxel de la c. Después de elegir a-x no se buscaron más pares.");
  image(s, "latent-interpolation-a-x.png", { x: X0, y: TOP - 0.1, w: XW, h: 4.6 }, 2.388);
  takeaway(s, "t = 0,5 genera un patrón nuevo a 1 píxel de la c · la salida binaria salta, no cambia de forma continua",
    C.orange, 6.25);
}

{
  const s = content("A", "Conclusión del ejercicio 1A", null,
    "Resumen: hace falta capacidad, la pérdida fue el cambio decisivo y repetir semillas cambió conclusiones varias veces.");
  columns(s, [
    { label: "Capacidad", color: C.orange, lines: ["Sin capas ocultas: 10-11 errores.",
      "Con BCE, hasta 15 neuronas nunca cumple.", "45 neuronas: 10 de 10."] },
    { label: "Función de pérdida", color: C.teal, lines: ["Con MSE el techo fue 2 de 3 semillas.",
      "BCE corrige los píxeles saturados.", "Fue el cambio decisivo."] },
    { label: "Semillas", color: C.blue, lines: ["A4 y D1 parecían buenas con una semilla.",
      "20 y 35 neuronas, con tres.", "Se eligió con 10."] },
  ], TOP + 0.1, 2.5);
  takeaway(s, "El latente separa las letras y permite generar, pero sus huecos no garantizan letras válidas",
    C.red, 4.75);
}

// ================================================================================
// 02 · Denoising autoencoder
divider("B", "02", "Denoising autoencoder",
  "Entrada con ruido bit-flip, salida limpia: cuánto ruido elimina la red en cinco niveles");

{
  const s = content("B", "Protocolo: medir el error antes y después de la red",
    "Ruido bit-flip: cada píxel se invierte con probabilidad p · ruido nuevo en cada época · evaluación con corrupciones nunca vistas",
    "Las letras de evaluación son las mismas 32; lo que nunca se vio es el ruido concreto. Todos los modelos de una comparación se evalúan sobre exactamente los mismos casos.");
  stat(s, "5", "niveles de ruido\np = 0 · 0,05 · 0,10 · 0,20 · 0,30", X0, TOP + 0.2, 3.7, C.teal);
  stat(s, "3.200", "casos por nivel y modelo\n100 corrupciones x 32 letras", X0 + 4.1, TOP + 0.2, 3.7, C.navy);
  stat(s, "3,5", "píxeles invertidos en promedio\ncon p = 0,10", X0 + 8.2, TOP + 0.2, 3.7, C.orange);
  table(s, ["Métrica", "Qué mide"], [
    ["Error de salida", "píxeles distintos entre la salida binarizada y la letra limpia"],
    ["Recuperación exacta", "porcentaje de casos en que la salida es idéntica a la letra limpia"],
    ["Casos empeorados", "porcentaje de casos que salen con más errores de los que entraron"],
  ], { y: 3.75, colW: [3.0, 8.93], rowH: 0.5, fontSize: 15 });
}

{
  const s = content("B", "La red del 1A, entrenada con ruido, empeoraba las letras",
    "35-45-2-45-35 · tanh · BCE · descenso de gradiente · entrenada y evaluada con p = 0,10 · semilla 0",
    "Con 2.000 épocas la salida tenía más errores que la entrada. Más épocas ayudaron poco. Con ruido el gradiente cambia en cada época, y la tasa 0,1 del 1A era demasiado grande. 0,03 fue mejor que 0,01 y 0,05 en los cinco niveles.");
  table(s, ["Épocas", "Tasa", "Error de entrada", "Error de salida", "Recuperación exacta"], [
    ["2.000", "0,1", "3,47", "3,58", "28,1 %"],
    ["10.000", "0,1", "3,47", "3,28", "34,0 %"],
    ["10.000", "0,03", "3,47", "1,30", "70,5 %"],
  ], { y: TOP, colW: [2.0, 1.6, 2.7, 2.7, 2.93], highlight: [2], rowH: 0.4, fontSize: 14, align: "center" });
  image(s, "ejs-figs/1b-d1-tasa.png", { x: X0, y: 3.35, w: XW, h: 3.0 }, 2.778);
  takeaway(s, "Bajar la tasa fue lo decisivo: 0,03 es la mejor en los cinco niveles (3 semillas: 73,4 ± 2,6 % exactas con p = 0,10)",
    C.teal, 6.38);
}

{
  const s = content("B", "Cómo se decide cada cambio del recorrido",
    "Una novedad por paso · misma evaluación para todas las alternativas · elegida en azul en cada gráfico",
    "Estas reglas se aplicaron en los pasos siguientes. Los niveles más informativos son 0,10 y 0,20: con 0,05 todo es casi perfecto y 0,30 es extremo.");
  steps(s, [
    ["Mirar la curva completa", "p = 0,10 y 0,20 son los más informativos"],
    ["Dos señales", "error de salida y recuperación exacta"],
    ["El control limpio no puede empeorar", "p = 0 debe quedar en 100 %"],
    ["Una semilla descarta, tres aceptan", "barras de error = desvío entre semillas"],
    ["Empate: la opción más simple", "menos parámetros o menos cómputo"],
  ], C.teal, TOP + 0.15, 0.78);
}

const dae = [
  ["Cuatro corrupciones por letra no estabilizan: se mantiene K = 1",
    "35-45-2-45-35 · BCE · GD 0,03 · 10.000 épocas · p_train = 0,10 · 3 semillas",
    "1b-d2-corrupciones.png", "K = 4 empeora y multiplica por 30 el desvío: con p = 0,10, 2,58 ± 1,39 contra 1,26 ± 0,04 errores",
    "La hipótesis era que promediar cuatro versiones daría un gradiente más estable. Fue al revés y además cuesta cuatro veces más ejemplos por época."],
  ["Comprimir en dos pasos limpia mejor que ensanchar",
    "BCE · GD 0,03 · 10.000 épocas · p_train = 0,10 · base y profunda con 3 semillas, ancha con semilla 0",
    "1b-d3-arquitectura.png", "Ancha y profunda tienen casi los mismos parámetros (4.597 y 4.717): gana la profunda desde p = 0,10",
    "Con tres semillas la profunda sube la recuperación de 73,4 % a 79,9 % con p = 0,10 y de 37,8 % a 49,8 % con p = 0,20."],
  ["Agregar letras limpias al lote dejó de estropear entradas sanas",
    "35-45-15-2-15-45-35 · BCE · GD 0,03 · 10.000 épocas · 32 limpias + 32 con ruido por época · 3 semillas",
    "1b-d4-lote.png", "Con p = 0: 90,6 % → 99,0 % exactas · con p = 0,10: 79,9 % → 88,4 %",
    "Con p = 0,10 solo el 2,5 % de las corrupciones deja la letra intacta, así que la red casi nunca veía letras limpias."],
  ["Tasa decreciente: igual media, menos variabilidad",
    "lote limpio + ruidoso · GD de 0,03 a 0,003 exponencial en 10.000 épocas · 3 semillas",
    "1b-d5-tasa-decreciente.png", "Desvío del error con p = 0,10 entre semillas: 0,112 → 0,002 píxeles",
    "Es la decisión con menos diferencia en la media. Se eligió porque el resultado dejó de depender de la semilla."],
  ["Latente 5: más espacio para separar las letras y su ruido",
    "lote limpio + ruidoso · tasa decreciente · latente 2 contra 5 · 3 semillas",
    "1b-d6-latente.png", "Error con p = 0,10: 0,66 → 0,40 píxeles, con solo 93 parámetros más",
    "La dimensión 2 era una exigencia del 1A. En el 1B se pide la arquitectura más conveniente para limpiar ruido."],
  ["Entrenar con varios niveles mejora incluso p = 0,30, que no se usó",
    "latente 5 · lote limpio + p = 0,05 + 0,10 + 0,20 contra solo 0,10 · 3 semillas",
    "1b-d7-ruido-entrenamiento.png", "Con p = 0,20: 53,0 % → 57,1 % exactas · el lote pasa de 64 a 128 ejemplos",
    "No se agregó p = 0,30 para no entrenar con letras casi irreconocibles. Esa variante se prueba al final."],
  ["Leaky ReLU mejora los niveles altos de ruido",
    "multinivel · tanh contra Leaky ReLU (pendiente 0,01) en las capas ocultas · 3 semillas",
    "1b-d8-activacion.png", "Con p = 0,20: 57,1 % → 64,4 % · con p = 0,30: 21,2 % → 28,0 %",
    "tanh se satura para valores grandes; Leaky ReLU no se satura del lado positivo."],
  ["Adam con tasa 0,005 supera al descenso de gradiente",
    "multinivel · Leaky ReLU · tamiz de 0,001 a 0,005 con semilla 0 · confirmación de 0,003 y 0,005 con 3 semillas",
    "1b-d10-tasa-adam.png", "Con una semilla, Adam superó al descenso; con tres, la tasa 0,005 gana en p = 0,10, 0,20 y 0,30",
    "Con ruido el gradiente cambia en cada época; Adam adapta el paso de cada peso. Las barras de 0,003 y 0,005 no se superponen en los niveles altos."],
];
for (const [title, subtitle, figure, message, notes] of dae) {
  const s = content("B", title, subtitle, notes);
  image(s, `ejs-figs/${figure}`, { x: X0, y: TOP - 0.05, w: XW, h: 4.45 }, 2.778);
  takeaway(s, message, C.teal, 6.2);
}

{
  const s = content("B", "Modelo congelado del ejercicio 1B",
    "Cada elemento salió de una comparación del recorrido · confirmado con 10 semillas sin reajustar",
    "Profunda y latente 5 para limpiar mejor; lote con letras limpias y varios niveles de ruido; Leaky ReLU y Adam 0,005.");
  columns(s, [
    { label: "Red", color: C.teal, lines: ["35 → 45 → 15 → 5 → 15 → 45 → 35", "4.810 parámetros",
      "Leaky ReLU (0,01) en las ocultas", "lineal en el latente, logística en la salida"] },
    { label: "Entrenamiento", color: C.blue, lines: ["BCE · Adam, tasa 0,005", "10.000 épocas",
      "por época: 32 limpias + 32 con p = 0,05", "+ 32 con p = 0,10 + 32 con p = 0,20"] },
    { label: "Por qué", color: C.orange, lines: ["profunda: comprime en dos pasos", "latente 5: más lugar por letra",
      "lote mixto: no estropea lo limpio", "Adam: gradiente cambiante"] },
  ], TOP + 0.25, 3.0);
}

{
  const s = content("B", "El DAE elimina el ruido de forma confiable hasta p = 0,10",
    "Modelo congelado · 10 semillas · 3.200 casos por nivel y semilla · banda = desvío entre semillas",
    "La banda es tan fina que casi no se ve: el resultado no depende de la semilla. Con p = 0,30 un tercio de los píxeles está invertido y muchas letras ya se parecen más a otra.");
  image(s, "dae/final/noise-curve-aggregate.png", { x: X0, y: TOP + 0.3, w: 8.5, h: 4.4 }, 2.487);
  sidebar(s, [
    { label: "p ≤ 0,10", head: "96,1 % exactas", body: "Error de 3,5 a 0,15 píxeles.", color: C.teal },
    { label: "p = 0,20", head: "74,7 % exactas", body: "Error de 7,0 a 1,7 píxeles.", color: C.orange },
    { label: "p = 0,30", head: "36,6 % exactas", body: "Error de 10,5 a 6,1; 17 % empeora.", color: C.red },
  ]);
}

{
  const s = content("B", "Los fallos aparecen cuando la letra se parece más a otra",
    "Casos elegidos con reglas fijadas antes de mirar · dificultad por carácter con p = 0,20 (10 semillas x 100 corrupciones)",
    "Fácil: f con 13 errores sale perfecta. Difícil: z mejora de 9 a 4. Fallido: d con 12 errores termina como una h, con 14. DEL es muy distinta del resto; |, `, u, r y n se confunden más.");
  image(s, "dae/final/representative-cases.png", { x: X0, y: TOP - 0.1, w: 4.1, h: 5.25 }, 0.79);
  image(s, "ejs-figs/1b-dificultad-por-letra.png", { x: 5.1, y: TOP, w: 7.53, h: 3.4 }, 2.273);
  text(s, "DEL se recupera el 96 % de las veces; | solo el 54 %. La d fallida termina dibujada como una h.",
    { x: 5.1, y: 5.35, w: 7.53, h: 0.9, fontSize: 15, color: C.navy, bold: true, align: "center" });
}

{
  const s = content("B", "Entrenar con p = 0,30: menos daño, menos letras perfectas",
    "Modelo final contra el mismo modelo + un bloque con p = 0,30 en el lote · 10 semillas · mismos casos de evaluación",
    "Con p = 0,30 el error baja de 6,06 a 5,85 y los casos empeorados de 16,9 % a 14,2 %, pero las letras perfectas bajan de 36,6 % a 34,9 %. Es un trade-off; se mantiene el modelo final.");
  image(s, "ejs-figs/1b-entrenar-con-p030.png", { x: X0, y: TOP - 0.05, w: XW, h: 4.45 }, 2.841);
  takeaway(s, "Menos errores y menos casos empeorados, pero menos letras perfectas: se mantiene el modelo final",
    C.teal, 6.2);
}

{
  const s = content("B", "Entrenar con ruido convierte puntos aislados en regiones",
    "Variante con latente 2 solo para poder dibujar · 100 corrupciones por letra con p = 0,10 · azul: recuperada exacta",
    "Izquierda: el autoencoder del 1A manda las letras con ruido a cualquier lado. Derecha: el DAE las agrupa en rayos que el decoder dibuja como la letra limpia. Los errores se concentran en el centro, donde se tocan las regiones.");
  image(s, "ejs-figs/1b-latente-ae-vs-dae.png", { x: X0, y: TOP - 0.1, w: XW, h: 4.6 }, 2.097);
  takeaway(s, "Recuperación con p = 0,10: 22,6 % en el autoencoder del 1A contra 88,8 % en el DAE", C.teal, 6.3);
}

{
  const s = content("B", "Resumen del recorrido del DAE",
    "Recuperación exacta acumulada después de cada decisión aceptada · media ± desvío entre semillas",
    "Los saltos más grandes los dieron la red profunda, el lote con letras limpias, Leaky ReLU y Adam. La tasa decreciente no mueve la media: aportó estabilidad.");
  image(s, "ejs-figs/1b-evolucion.png", { x: X0, y: TOP - 0.1, w: XW, h: 4.6 }, 2.115);
  takeaway(s, "Con p = 0,10: de 73 % a 96 % · con p = 0,20: de 38 % a 75 %", C.teal, 6.3);
}

{
  const s = content("B", "Conclusión del ejercicio 1B", null,
    "El DAE elimina ruido de forma muy confiable hasta p = 0,10, con fallas en 0,20 y solo mejora el promedio en 0,30. El límite no es solo de la red: con mucho ruido la letra se parece más a otra.");
  columns(s, [
    { label: "Entrenamiento", color: C.teal, lines: ["Con ruido, la tasa del 1A era demasiado grande.",
      "Letras limpias y varios niveles en el lote."] },
    { label: "Arquitectura", color: C.blue, lines: ["Profunda y con latente 5.",
      "La red del 1A estaba pensada para graficar, no para limpiar."] },
    { label: "Resultado", color: C.orange, lines: ["96 % exactas con p = 0,10.",
      "75 % con 0,20 y 37 % con 0,30.", "Casi idéntico entre semillas."] },
  ], TOP + 0.1, 2.5);
  takeaway(s, "Entrenar con ruido llena el espacio alrededor de cada letra: el VAE lo hará a propósito", C.red, 4.75);
}

// ================================================================================
// 03 · VAE
divider("C", "03", "Autoencoder variacional",
  "Generar funciones de Python nuevas que pertenezcan a la familia del dataset");

{
  const s = content("C", "Dataset: funciones de Python verificables",
    "Plantilla fija de 26 caracteres · operadores {+, −, *} · constantes de 0 a 99 · splits balanceados por par de operadores",
    "Elegimos código porque la pertenencia se puede verificar objetivamente. Cada carácter se codifica one-hot sobre 25 símbolos y la salida aplica un softmax por posición con entropía cruzada.");
  text(s, "def f(x): return (x+47)*83", { x: X0, y: TOP + 0.1, w: 6.0, h: 0.6, fontSize: 24,
    fontFace: MONO, bold: true, color: C.navy });
  stat(s, "90.000", "funciones posibles", X0, TOP + 1.0, 2.9, C.blue);
  stat(s, "650", "entradas por función\n26 posiciones x 25 símbolos", X0 + 3.1, TOP + 1.0, 2.9, C.navy);
  stat(s, "1.800", "entrenamiento", X0, TOP + 2.75, 1.9, C.teal);
  stat(s, "450", "validación", X0 + 2.05, TOP + 2.75, 1.9, C.teal);
  stat(s, "450", "test", X0 + 4.1, TOP + 2.75, 1.9, C.teal);
  sidebar(s, [
    { label: "1 · Compila", head: "Python la acepta", body: "Se compila sin ejecutarla.", color: C.blue },
    { label: "2 · Pertenece", head: "Respeta la plantilla", body: "Operadores y constantes válidos.", color: C.teal },
    { label: "3 · Nueva en texto", head: "No está en el dataset", body: "Ninguna de las 2.700.", color: C.orange },
    { label: "4 · Calcula algo nuevo", head: "f(x) = a·x + b nueva", body: "Su par (a, b) no aparece.", color: C.navy },
  ], 7.6, TOP + 0.05, XW + X0 - 7.6);
}

{
  const s = content("C", "Control: un autoencoder común reconstruye el 99,6 % del test",
    "latente 16 · tanh · softmax por posición · entropía cruzada · Adam 0,005 · hasta 3.000 épocas · semilla 0",
    "Regla prefijada: 100 % exacto en entrenamiento, al menos 99 % en validación y 100 % de gramática en tres controles seguidos. El test se abrió una sola vez, con la red elegida.");
  table(s, ["Arquitectura", "Parámetros", "Train exacto", "Validación exacta", "Pasa"], [
    ["650-64-16-64-650", "86.042", "100 %", "97,33 %", "no"],
    ["650-128-16-128-650", "171.418", "100 %", "95,78 %", "no"],
    ["650-256-16-256-650", "342.170", "100 %", "99,11 %", "sí"],
  ], { colW: [3.4, 2.1, 2.1, 2.5, 1.83], highlight: [2], rowH: 0.6, fontSize: 16 });
  stat(s, "448 / 450", "funciones de test reconstruidas exactas", X0, 4.4, XW, C.teal);
  takeaway(s, "Si el VAE falla, no es por capacidad de la red ni por la representación", C.navy, 6.1);
}

{
  const s = content("C", "Referencia: qué generaría un generador ideal",
    "Sortear 1.000 funciones uniformemente de las 90.000 (como se armó el dataset) · promedio de 50 sorteos",
    "El dataset es una muestra uniforme de la familia, así que el generador ideal es uno uniforme. Un modelo con más novedad que este no inventa más: genera de más ciertos tipos de función.");
  stat(s, "100 %", "válidas", X0, TOP + 0.2, 2.25, C.teal);
  stat(s, "96,8 %", "nuevas en texto", X0 + 2.4, TOP + 0.2, 2.25, C.blue);
  stat(s, "55,0 %", "calculan algo nuevo", X0 + 4.8, TOP + 0.2, 2.25, C.blue);
  stat(s, "0,033", "distancia en pares\nde operadores", X0 + 7.2, TOP + 0.2, 2.25, C.navy);
  stat(s, "0,125", "distancia en\nconstantes", X0 + 9.6, TOP + 0.2, 2.25, C.navy);
  text(s, "Regla de elección", { x: X0, y: 3.75, w: 4, h: 0.4, fontSize: 17, bold: true, color: C.navy });
  steps(s, [
    ["Requisitos en todas las semillas", "≥ 90 % de reconstrucción · ≥ 99 % de muestras válidas"],
    ["Gana la distribución más parecida a los datos", "distancia de variación total a la uniforme"],
    ["Empate dentro del desvío", "el latente más chico"],
  ], C.blue, 4.25, 0.66);
}

{
  const s = content("C", "Paso 1 · beta regula reconstruir contra generar",
    "650-256-16-256-650 · Adam 0,005 · 3.000 épocas · semilla 0 · 1.000 muestras fijas de N(0, I)",
    "Sin KL solo el 28,6 % de lo generado es válido: el latente tiene huecos. Con beta 1 la reconstrucción cae al 38 %. beta 0,03 genera 99 % válido pero con sesgo en los pares de operadores.");
  image(s, "ejs-figs/2-paso1-beta.png", { x: X0, y: TOP - 0.05, w: XW, h: 4.45 }, 2.75);
  takeaway(s, "Sin KL: 28,6 % válidas · beta 1: 38 % reconstruido · beta 0,1 y 0,3 quedan cerca del ideal", C.blue, 6.2);
}

{
  const s = content("C", "Paso 2 · el latente necesita al menos 16 dimensiones",
    "650-256-z-256-650 · beta 0,03 / 0,1 · Adam 0,005 · 3.000 épocas · semilla 0",
    "Con 2 dimensiones no entra la información de una función: genera válido pero repite pocas funciones. Con 8 reconstruye pero sesga las constantes. 16 y 32 cumplen las tres preguntas.");
  table(s, ["Latente", "¿Reconstruye? (validación)", "¿Genera válido?", "¿Como los datos? (dist. constantes)", "Veredicto"], [
    ["2", "0,2 % / 0,7 %", "100 % / 100 %", "0,57 / 0,51", "no reconstruye"],
    ["8", "99,8 % / 92,4 %", "95,1 % / 100 %", "0,28 / 0,28", "sesgado"],
    ["16", "100 % / 100 %", "99,4 % / 100 %", "0,20 / 0,17", "pasa"],
    ["32", "100 % / 100 %", "100 % / 100 %", "0,14 / 0,15", "pasa, más cerca del ideal"],
  ], { colW: [1.2, 2.9, 2.3, 3.2, 2.33], highlight: [3], rowH: 0.62, fontSize: 15, align: "center" });
  text(s, "Cada celda: beta 0,03 / beta 0,1 · distancia del generador ideal: 0,125", { x: X0, y: 5.05,
    w: XW, h: 0.35, fontSize: 12, color: C.gray, align: "center" });
  takeaway(s, "Pasan al paso 3: beta 0,1 y 0,3 con latente 16; beta 0,03 y 0,1 con latente 32", C.blue, 5.75);
}

{
  const s = content("C", "Paso 3 · beta 0,03 con latente 32 genera más parecido a los datos",
    "4 finalistas · 3 semillas · barra = media, puntos = semillas, línea = generador ideal",
    "Los cuatro finalistas cumplen los requisitos en las tres semillas. En pares de operadores todos quedan en el ruido del ideal. En constantes, beta 0,03 con latente 32 es la más cercana y la más estable.");
  image(s, "ejs-figs/2-paso3-finalistas.png", { x: X0, y: TOP - 0.05, w: XW, h: 4.3 }, 3.125);
  takeaway(s, "Distancia en constantes: 0,144 ± 0,001 contra 0,153-0,184 de los demás · ideal 0,125", C.blue, 6.1);
}

{
  const s = content("C", "Modelo congelado del ejercicio 2",
    "Confirmado con 10 semillas sin reajustar · el test se evaluó una sola vez, recién acá",
    "La reconstrucción baja rápido y después sigue con ruido porque cada época sortea un epsilon distinto. La KL sube al principio y la regularización la estabiliza en 22.");
  columns(s, [
    { label: "Red", color: C.blue, lines: ["650 → 256 → (μ, log σ²) 32 → 256 → 650", "tanh en las ocultas",
      "softmax por posición en la salida"] },
    { label: "Entrenamiento", color: C.teal, lines: ["entropía cruzada + 0,03 · KL", "Adam 0,005 · 3.000 épocas",
      "full-batch con 1.800 funciones"] },
  ], TOP + 0.1, 1.9);
  image(s, "ejs-figs/2-perdidas.png", { x: X0, y: 3.85, w: XW, h: 2.95 }, 2.895);
}

{
  const s = content("C", "Con 10 semillas, genera como un generador ideal de la familia",
    "Media ± desvío de 10 semillas · 1.000 muestras de N(0, I) por semilla",
    "Reconstruye todo el test y todo lo generado es válido. Novedad y diversidad coinciden con las del ideal. La única diferencia medible es una leve preferencia en las constantes.");
  stat(s, "100 %", "test reconstruido exacto", X0, TOP + 0.2, 3.8, C.teal);
  stat(s, "100 %", "muestras válidas", X0 + 4.05, TOP + 0.2, 3.8, C.teal);
  stat(s, "993,5", "funciones distintas cada 1.000", X0 + 8.1, TOP + 0.2, 3.8, C.navy);
  table(s, ["Medida", "VAE (10 semillas)", "Generador ideal"], [
    ["Nuevas en texto", "96,4 % ± 0,6", "96,8 %"],
    ["Calculan algo nuevo", "54,4 % ± 1,5", "55,0 %"],
    ["Distancia en pares de operadores", "0,043 ± 0,011", "0,033"],
    ["Distancia en constantes", "0,145 ± 0,006", "0,125"],
  ], { y: 3.6, colW: [5.0, 3.5, 3.43], rowH: 0.5, fontSize: 15 });
}

{
  const s = content("C", "Muestras generadas: las primeras 12, sin elegir",
    "z ~ N(0, I) → decoder → carácter más probable en cada posición · semilla 0",
    "Para generar no se usa el encoder: se sortea z y se decodifica. Las 12 pertenecen a la familia y 11 no están en el dataset. Que sea nueva en texto pero no en lo que calcula es esperable: (x-89)+ 8 calcula lo mismo que (x-81)+ 0.");
  const rows = [
    ["(x+10)*88", "sí", "sí"], ["(x*74)+30", "no", "no"], ["(x-89)+ 8", "sí", "no"],
    ["(x-93)*31", "sí", "sí"], ["(x+14)-58", "sí", "no"], ["(x*85)-13", "sí", "sí"],
    ["(x-69)*18", "sí", "sí"], ["(x* 5)*59", "sí", "no"], ["(x*35)+71", "sí", "sí"],
    ["(x- 6)-90", "sí", "no"], ["(x+19)*32", "sí", "sí"], ["(x+97)+48", "sí", "no"],
  ];
  const half = (part) => part.map((r, i) => [`def f(x): return ${r[0]}`, r[1], r[2]]);
  table(s, ["Función generada", "Nueva", "Calcula algo nuevo"], half(rows.slice(0, 6)),
    { x: X0, w: 5.8, colW: [3.4, 1.0, 1.4], rowH: 0.5, fontSize: 13, fontFace: MONO });
  table(s, ["Función generada", "Nueva", "Calcula algo nuevo"], half(rows.slice(6)),
    { x: X0 + 6.13, w: 5.8, colW: [3.4, 1.0, 1.4], rowH: 0.5, fontSize: 13, fontFace: MONO });
  takeaway(s, "12 de 12 pertenecen a la familia · 11 no están en el dataset · 6 calculan funciones nuevas", C.blue, 5.6);
}

{
  const s = content("C", "La KL hace que el prior caiga donde están los datos",
    "Proyección PCA de los códigos, solo para visualizar · gris: funciones de entrenamiento · color: 1.000 z ~ N(0, I)",
    "Sin KL los datos ocupan una región mucho más grande que el prior y el 71 % de lo sorteado sale inválido. Con KL los datos quedan sobre la misma región que el prior y todo lo sorteado es válido.");
  image(s, "ejs-figs/2-latente-prior.png", { x: X0, y: TOP - 0.1, w: XW, h: 4.65 }, 2.222);
  takeaway(s, "Desvío de los códigos: 4,9 sin KL contra 0,86 con KL · muestras válidas: 28,6 % contra 100 %", C.blue, 6.3);
}

{
  const s = content("C", "Interpolación entre dos funciones: todo el recorrido es válido",
    "z(t) = (1 − t) · z_a + t · z_b entre las dos primeras muestras que calculan algo nuevo · semilla 0",
    "Se muestran solo los t donde cambia la función. En cada paso cambia una sola parte: una constante, un operador u otra constante. En el autoencoder del 1A la interpolación pasaba por patrones que no eran letras.");
  table(s, ["t", "Función decodificada", "¿En el dataset?"], [
    ["0,0", "def f(x): return (x+10)*88", "no"],
    ["0,3", "def f(x): return (x+13)*36", "no"],
    ["0,4", "def f(x): return (x+93)*36", "no"],
    ["0,7", "def f(x): return (x-93)*36", "no"],
    ["0,8", "def f(x): return (x-93)*30", "no"],
    ["1,0", "def f(x): return (x-93)*31", "no"],
  ], { x: X0 + 1.5, w: XW - 3.0, colW: [1.2, 5.93, 1.8], rowH: 0.52, fontSize: 16, fontFace: MONO });
  takeaway(s, "Los 11 puntos del recorrido son funciones válidas y ninguna está en el dataset", C.blue, 5.75);
}

{
  const s = content("C", "Conclusión del ejercicio 2", null,
    "La KL es lo que hace posible generar. beta regula el equilibrio y hace falta un latente suficientemente grande. Medir contra un generador ideal evitó confundir sesgo con creatividad.");
  columns(s, [
    { label: "La KL", color: C.blue, lines: ["Sin KL: 28,6 % válidas.", "Con beta 0,03: 100 %.",
      "Lleva los datos a cubrir el prior."] },
    { label: "beta y latente", color: C.teal, lines: ["beta 0,01 deja huecos.", "beta 1 reconstruye el 38 %.",
      "Hace falta latente ≥ 16."] },
    { label: "Resultado", color: C.orange, lines: ["100 % del test reconstruido.", "100 % de muestras válidas.",
      "Novedad igual a la del ideal."] },
  ], TOP + 0.1, 2.5);
  takeaway(s, "Más novedad no es mejor: un exceso de funciones nuevas indicaba un sesgo del modelo", C.red, 4.75);
}

// ================================================================================
// Cierre
{
  const s = content("Z", "Conclusiones", null,
    "Tres modelos, la misma idea: el cuello de botella obliga a aprender la estructura de los datos. Lo que cambia es qué tan controlado queda el espacio latente.");
  columns(s, [
    { label: "Ejercicio 1A", color: C.orange, lines: ["35-45-2-45-35 con BCE reconstruye los 32 patrones en 10 de 10 semillas.",
      "El latente separa las letras, pero tiene huecos."] },
    { label: "Ejercicio 1B", color: C.teal, lines: ["El DAE recupera el 96 % de las letras con p = 0,10.",
      "Entrenar con ruido convierte puntos en regiones."] },
    { label: "Ejercicio 2", color: C.blue, lines: ["El VAE genera funciones válidas y nuevas como un generador ideal.",
      "La KL llena el espacio latente."] },
  ], TOP + 0.1, 2.5);
  text(s, "Conclusión transversal", { x: X0, y: 4.55, w: XW, h: 0.4, fontSize: 17, bold: true, color: C.navy,
    align: "center" });
  takeaway(s, "Del punto aislado a la región: cuanto más controlado el latente, más confiable es generar", C.red, 5.0);
}

pres.writeFile({ fileName: OUT }).then((file) => console.log(`escrito: ${file}`));
