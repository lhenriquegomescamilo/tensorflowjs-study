$('#result').text('');

const tensorEixoX = tf.tensor([2,  4,  6,  8, 10, 12, 14,16,18]);
const tensorEixoY = tf.tensor([30, 25, 22, 18, 15]);

function exibir(str = '') {
    $('#result').text(str);

}

function tensorToArray(tensor) {
    return Array.from(tensor.dataSync());
}

function arrayToTensor(array) {
    return tf.tensor(array);
}

function regressaoLinear(arrayDoEixoX, arrayDoEixoY, vetor) {
    const x = arrayToTensor(arrayDoEixoX);
    const y = arrayToTensor(arrayDoEixoY);

    const resultadoPrimeiraEtapa = x.sum().mul(y.sum()).div(x.size);
    const resultadoSegundoEtapa = x.sum().mul(x.sum()).div(x.size);
    const resultadoTerceiraEtapa = x.mul(y).sum().sub(resultadoPrimeiraEtapa);
    const resultadoQuartaEtapa = resultadoTerceiraEtapa.div(x.square().sum().sub(resultadoSegundoEtapa));
    const resultadoQuintaEtapa = y.mean().sub(resultadoQuartaEtapa.mul(x.mean()));

    // debugger;
    const tensor = resultadoQuartaEtapa.mul(vetor).add(resultadoQuintaEtapa);
    return tensorToArray(tensor);
}

function executar() {
    let txt = '';
    const vetorEixoX = tensorToArray(tensorEixoX);
    const vetorEixoY = tensorToArray(tensorEixoY);

    const tamanhoEixoX = vetorEixoX.length;
    const tamanhoEixoY = vetorEixoY.length;

    const vetorTemporarioEixoXComMesmoTamanhoDoEixoX = vetorEixoX.slice(0, tamanhoEixoY);
    const vetorTemporarioEixoY = vetorEixoY;

    const diferencaEntreEixoYehEixoX = tamanhoEixoX - tamanhoEixoY;

    if (diferencaEntreEixoYehEixoX > 0) {
        const regressao = [];
        for (let i = 0; i < diferencaEntreEixoYehEixoX; i++) {
            const resultadoRegressaoLinear = regressaoLinear(vetorTemporarioEixoXComMesmoTamanhoDoEixoX, vetorTemporarioEixoY, vetorEixoX[tamanhoEixoY + i])
            resultadoRegressaoLinear.forEach(resultado => regressao.push(resultado))
        }
        const valorDoNovoYEncontrado = vetorTemporarioEixoY.concat(regressao);
        const tensorResultante = tf.tensor(valorDoNovoYEncontrado);

        txt += `Regrassão Lienar Simples\n
				ANTES: \n\n
				${tensorEixoX.toString()}
				${tensorEixoY.toString()}
				\n\n
				DEPOIS\n\n
				${tensorResultante.toString()}
				`;
    }

    exibir(txt);
}
