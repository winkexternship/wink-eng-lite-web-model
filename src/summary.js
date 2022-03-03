var summary = function ( rdd, its ) {

    var tokens = rdd.tokens;
    var sentences = rdd.sentences;
    var cache = rdd.cache;
    var numOfSentences = rdd.sentences.length;
    var aptTokens = [];
    var aptPOS = [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];
    var weights = [];
    var convergenceThreshold =  0.0001;
    var maxIteration = 100;
    var dampingFactor = 0.85;
    var numOfConvergence = 0;
    var colSum = [];
    var nonZero = [];
    var ranks = [];

    console.log(`Number of sentences in text:${numOfSentences}`);
    // Getting appropriate tokens for text summarization (Preprocessing)
    for ( let i = 0; i < numOfSentences; i += 1) {
        const temp = [];
        for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1) {
            if ( its.type( j, tokens, cache ) === 'word' && !its.stopWordFlag( j, tokens, cache ) && aptPOS.includes(its.pos(j, tokens, cache)) ) {
                temp.push(its.normal( j, tokens, cache ));
            }
        }
        aptTokens.push(temp);
    }
    console.log(`Number of sentences in aptTokens:${aptTokens.length}`);
    console.log('aptTokens:');
    console.log(aptTokens);

    // Creating adjacency matrix for graph of sentences
    var senGraph = new Array(numOfSentences);
    for ( let i = 0; i < numOfSentences; i += 1) {
        senGraph[i] = new Array(numOfSentences);
    }

    // Populating the matrix with weights
    for ( let i = 0; i < numOfSentences; i += 1) {
        senGraph[i][i] = 0;
        for ( let j = i + 1; j < numOfSentences; j += 1) {
            // change the following to implement different ways to calculate simmilarity value
            const numOfCommonTokens = aptTokens[i].filter((token) => aptTokens[j].includes(token)).length;
            senGraph[i][j] = numOfCommonTokens / (Math.log(aptTokens[i].length) + Math.log(aptTokens[j].length));
            if( Number.isNaN (senGraph[i][j] ) )
                senGraph[i][j] = 0;
            senGraph[j][i] = senGraph[i][j];
        }
    }
    console.log('SenGraph:');
    console.log(senGraph);

    // Page Rank
    for ( let i = 0; i < numOfSentences; i += 1) {
        let c = 0;
        for ( let j = 0; j < numOfSentences; j += 1 ) {
            if ( senGraph[i][j] !== 0 ) c += 1;
        }
        nonZero.push( c );
    }

    for ( let i = 0; i < numOfSentences; i += 1 ) {
        let s = 0;
        for ( let j = 0; j < numOfSentences; j += 1 ) {
            if ( senGraph[i][j] !== 0)
                s += senGraph[i][j] / nonZero[j];
        }
        ranks.push({ idx: i, val: s });
        }



    // // Normalizing the matrix
    // for ( let i = 0; i < numOfSentences; i += 1) {
    //     colSum[i] = senGraph[i].reduce((previousValue, currentValue) =>  previousValue + currentValue ,0);
    // }
    // // for ( let i = 0; i < numOfSentences; i += 1) {
    // //     for ( let j = rdd, its.value, as.text, addons0; j < numOfSentences; j += 1) {
    // //         senGraph[j][i] /= colSum[i];
    // //     }
    // // }

    // // Initializing the weights vector
    // for ( let i = 0; i < numOfSentences; i += 1) {
    //     weights.push({ idx: i, val: 1 / numOfSentences });
    // }
    // console.log('Intital Weights:');
    // console.log(weights);

    // // Ranking Algorithm (PageRank)
    // for ( let i = 0; i < maxIteration; i += 1) {
    //     for ( let j = 0; j < numOfSentences; j += 1) {
    //         let rank = 1 - dampingFactor;
    //         rank += dampingFactor * senGraph[j].reduce((previousValue, currentValue, idx) => previousValue + ((currentValue / colSum[idx]) * weights[idx].val) ,0);
    //         if ( Math.abs(weights[j] - rank) <= convergenceThreshold ) {
    //             numOfConvergence += 1;
    //         }
    //         weights[j].val = rank;
    //     }
    //     if ( numOfConvergence === numOfSentences ) {
    //         break;
    //     }
    // }

    // Sentence Selection
    ranks.sort( (a, b) => b.val - a.val );
    console.log('Final Weights:');
    console.log(ranks);
    
    return {
        weights: ranks,
        numOfSentences: numOfSentences
    };

    // return 'This is a summary text for the given input text.';
};

module.exports = summary;
