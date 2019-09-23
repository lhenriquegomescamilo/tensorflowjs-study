import * as fs from 'fs';
import * as path from 'path';
import * as tf from '@tensorflow/tfjs-node-gpu';
import { Logger } from '@nestjs/common';
import * as R from 'ramda';

const fnNoEmpty = R.compose(R.not, R.empty);
const logger = new Logger('tensorflow-reader-csv.ts');
const pathCSV = path.join(__dirname, '../assets/csv/USD_BRL_Dados_Historicos.csv');
const file = fs.readFileSync(pathCSV, { encoding: 'utf8' });
const fileCleaned = file.toString().trim();
const linhas = fileCleaned.split('\r\n');
const eixoX = [];
const eixoY = [];
const removeQuote = (stringValue) => {
  if (!stringValue) {
    return stringValue;
  }
  return stringValue.replace('"', '').replace('"', '');
};
let quantidadeLinhas = 0;
for (let indice = 1; indice < linhas.length; indice++) {
  try {
    let celula1 = [];
    const lineCelular1 = linhas[indice + 1].split(',');
    if (lineCelular1 && lineCelular1.every(fnNoEmpty)) {
      celula1 = lineCelular1;
    }
    const celula2 = linhas[indice].split(',');
    const FechamentoX = Number(removeQuote(celula1[1]));
    const AberturaX = Number(removeQuote(celula1[2]));
    const MaximaX = Number(removeQuote(celula1[3]));
    const MinimaX = Number(removeQuote(celula1[4]));
    eixoX.push([FechamentoX, AberturaX, MaximaX, MinimaX]);

    const FechamentoY = Number(removeQuote(celula2[1]));
    const AberturaY = Number(removeQuote(celula2[2]));
    const MaximaY = Number(removeQuote(celula2[3]));
    const MinimaY = Number(removeQuote(celula2[4]));
    eixoY.push([FechamentoY, AberturaY, MaximaY, MinimaY]);
    // }
    // if (celula1NaoPossuiValoresVazios && celulas2NaoPossuiValoresVazios) {
    quantidadeLinhas++;
    // }
  } catch (error) {
    logger.error(error, `erro em ${indice}`);
  }
}

const model = tf.sequential({
  layers: [
    tf.layers.dense({ inputShape: [4], units: 4 }),
  ],
});
// const inputLayer = tf.layers.dense({ units: 4, inputShape: [4] });
// model.add(inputLayer);
model.compile({
  optimizer: 'sgd',
  loss: 'meanSquaredError',
  metrics: ['accuracy'],
});
const x = tf.tensor(eixoX, [quantidadeLinhas, 4]);
const y = tf.tensor(eixoY);

// const arrInput = [[3.8735, 3.9202, 3.9456, 3.8685]]; // 27.12.2018
// const arrInput2 = [["3.8813","3.8769","3.8956","3.8298"]]; // 31.12.2018
// 28.12.2018,"3.8813","3.8769","3.8956","3.8298"
const arrInput = [[3.8813, 3.8769, 3.8956, 3.8298]];

const input = tf.tensor(arrInput, [1, 4]);

model.fit(x, y, { epochs: 800 }).then(() => {
  // @ts-ignore
  const output = model.predict(input).dataSync();
  logger.log('PRECO DAS COTACOES');
  logger.log(`Fechamento: R$ ${Number(output[0].toFixed(4))}`);
  logger.log(`Abertura:   R$ ${Number(output[1].toFixed(4))}`);
  logger.log(`Maximo:     R$ ${Number(output[2].toFixed(4))}`);
  logger.log(`Minimo: R$ ${Number(output[3].toFixed(4))}`);
});
// }).catch(error => logger.error(error));
