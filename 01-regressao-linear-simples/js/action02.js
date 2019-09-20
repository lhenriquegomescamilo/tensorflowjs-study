$('#eixoX').text('');
$('#eixoY').text('');

function executar() {
    const eixoX = $('#eixoX').val();
    const eixoY = $('#eixoY').val();

    const arrayEixoX = eixoX.split(',');
    const arrayEixoY = eixoY.split(',');

    const vetorX = converteNumero(arrayEixoX);
    const vetorY = converteNumero(arrayEixoY);

    const tamanhoEixoX = vetorX.length;
    const tamanhoEixoY = vetorY.length;

    const vetorTemporarioEixoXComMesmoTamanhoDoEixoX = vetorX.slice(0, tamanhoEixoY);
    const vetorTemporarioEixoY = vetorY;

    const diferencaEntreEixoYehEixoX = tamanhoEixoX - tamanhoEixoY;

    if (diferencaEntreEixoYehEixoX > 0) {
        const regressao = [];
        for (let i = 0; i < diferencaEntreEixoYehEixoX; i++) {
            const resultadoRegressaoLinear = regressaoLinear(
                vetorTemporarioEixoXComMesmoTamanhoDoEixoX,
                vetorTemporarioEixoY,
                vetorX[tamanhoEixoY + i]
            );
            resultadoRegressaoLinear.forEach(resultado => regressao.push(resultado))
        }
        const valorDoNovoYEncontrado = vetorTemporarioEixoY.concat(regressao);
        // const tensorResultante = tf.tensor(valorDoNovoYEncontrado);
        $('#eixoY').val(valorDoNovoYEncontrado.toString().replace(/,/g, ', '));
    }
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

function tensorToArray(tensor) {
    return Array.from(tensor.dataSync());
}

function arrayToTensor(array) {
    return tf.tensor(array);
}

function converteNumero(array) {
    const temp = [];
    for (let i = 0; i < array.length; i++) {
        temp.push(parseFloat(array[i].toString().trim()));
    }
    return temp;
}
