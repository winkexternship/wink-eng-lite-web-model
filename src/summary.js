var summary = function ( rdd, its ) {

    var tokens = rdd.tokens;
    var sentences = rdd.sentences;
    var cache = rdd.cache;
    var numOfSentences = rdd.sentences.length;
    var aptTokens = [];
    var aptPOS = [ 'ADJ', 'ADV', 'NOUN', 'PROPN', 'VERB' ];

    // Getting appropriate tokens for text summarization
    for ( let i = 0; i < numOfSentences; i += 1) {
        var temp = [];
        for ( let j = sentences[i][0]; j <= sentences[i][1]; j += 1) {
            if ( its.type( j, tokens, cache ) === 'word' && aptPOS.includes(its.pos(j, tokens, cache)) ) {
                temp.push(its.normal( j, tokens, cache ));
            }
        }
        aptTokens.push(temp);
    }

    // Creating adjacency matrix for graph of sentences
    var senGraph = new Array(numOfSentences);
    for ( let i = 0; i < numOfSentences; i += 1) {
        senGraph[i] = new Array(numOfSentences);
    }

    // Populating the matrix with weights
    for ( let i = 0; i < numOfSentences - 1; i += 1) {
        for ( let j = i + 1; j < numOfSentences; j += 1) {
            const numOfCommonTokens = aptTokens[i].filter((token) => aptTokens[j].includes(token)).length;
            senGraph[i][j] = numOfCommonTokens / (Math.log(aptTokens[i].length) + Math.log(aptTokens[j].length));
            senGraph[j][i] = senGraph[i][j];
        }
    }

    // Ranking the sentences

    return 'This is a summary text for the given input text.';
};

module.exports = summary;
