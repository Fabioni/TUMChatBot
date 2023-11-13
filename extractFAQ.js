// Function to extract questions and answers from https://www.mgt.tum.de/faq-center
function extractQA() {
    const qaList = [];
    // Select all the question elements
    const questionElements = document.querySelectorAll('h6[data-v-62390528]');

    questionElements.forEach(questionElement => {
        const question = questionElement.innerText.trim();
        // Find the corresponding answer element
        let answerElement = questionElement.closest("li").querySelector("[data-v-f13f6820][data-v-62390528]")
        const answer = answerElement ? answerElement.innerText.trim() : '';

        qaList.push({ question, answer });
    });

    return qaList;
}

// Run the function and log the results
const results = extractQA();
console.log(results);
