import OpenAI from 'openai'; 
import { Ingredient, DietaryPreference, Recipe } from '../types/index'
import aiGenerated from './models/aigenerated';
import { connectDB } from '../lib/mongodb';
import { ImagesResponse } from 'openai/resources';
import { HfInference } from '@huggingface/inference';

const openai = new OpenAI({
    apiKey: process.env.OPENAI_API_KEY,
});
const hf = new HfInference(process.env.HUGGINGFACE_API_KEY);


const saveOpenaiResponses = async ({ userId, prompt, response }: { userId: string, prompt: string, response: any }) => {
    try {
        await connectDB();
        const { _id } = await aiGenerated.create({
            userId,
            prompt,
            response,
        });
        return _id
    } catch (error) {
        console.error('Failed to save response to db:', error);
        return null
    }
}

const getRecipeGenerationPrompt = (ingredients: Ingredient[], dietaryPreferences: DietaryPreference[]) => `
I have the following ingredients: ${JSON.stringify(ingredients)} ${dietaryPreferences.length ? `and dietary preferences: ${dietaryPreferences.join(',')}` : ''}. Please provide me with three different delicious recipes. The response should be in the following JSON format without any additional text or markdown:
[
    {
        "name": "Recipe Name",
        "ingredients": [
            {"name": "Ingredient 1", "quantity": "quantity and unit"},
            {"name": "Ingredient 2", "quantity": "quantity and unit"},
            ...
        ],
        "instructions": [
            "Step 1",
            "Step 2",
            ...
        ],
        "dietaryPreference": ["Preference 1", "Preference 2", ...],
        "additionalInformation": {
            "tips": "Some cooking tips or advice.",
            "variations": "Possible variations of the recipe.",
            "servingSuggestions": "Suggestions for serving the dish.",
            "nutritionalInformation": "Nutritional information about the recipe."
        }
    },
    ...
]
Please ensure the recipes are diverse and use the ingredients listed. The recipes should follow the dietary preferences provided. The instructions should be ordered but not include the step numbers.
`;

const getImageGenerationPrompt = (recipeName: string, ingredients: Recipe['ingredients']): string => {
    const allIngredients = ingredients.map(ingredient => `${ingredient.name}`).join(', ');
    const prompt = `Create an image of a delicious ${recipeName} made of these ingredients: ${allIngredients}. The image should be visually appealing and showcase the dish in an appetizing manner.`;
    return prompt;
};

const getIngredientValidationPrompt = (ingredientName: string): string => {
    return `You are a food ingredient validation assistant. Given this ingredient name: ${ingredientName}, you will respond with a JSON object in the following format:

{
  "isValid": true/false,
  "possibleVariations": ["variation1", "variation2", "variation3"]
}

The "isValid" field should be true if the ingredient is commonly used in recipes and false otherwise. The "possibleVariations" field should be an array containing 2 or 3 variations or related ingredients to the provided ingredient name. If no variations or related ingredients are real and commonly used, return an empty array.

Do not include any Markdown formatting or code blocks in your response. Return only valid JSON.`
}

type ResponseType = {
    recipes: string | null
    openaiPromptId: string
}
// Define the response structure for Hugging Face API
interface HfResponse {
    generated_text: string;
}

// Update the generateRecipe function
export const generateRecipe = async (ingredients: Ingredient[], dietaryPreferences: DietaryPreference[], userId: string): Promise<ResponseType> => {
    try {
        const prompt = getRecipeGenerationPrompt(ingredients, dietaryPreferences);
        
        // Call the Hugging Face API to generate text
        const response= await hf.textGeneration({
            model: 'EleutherAI/gpt-neo-2.7B',
            inputs: prompt,
            parameters: {
                max_length: 1500,
                do_sample: true,
                top_k: 50,
                top_p: 0.95,
                num_return_sequences: 1,
            },
        });

        // Access the generated text from the response
        const generatedText = response.generated_text; // Now TypeScript knows the type

        const _id = await saveOpenaiResponses({ userId, prompt, response });

        return { recipes: generatedText, openaiPromptId: _id || 'null-prompt-id' };
    } catch (error) {
        console.error('Failed to generate recipe:', error);
        throw new Error('Failed to generate recipe');
    }
};


// // export const generateRecipe = async (ingredients: Ingredient[], dietaryPreferences: DietaryPreference[], userId: string): Promise<ResponseType> => {
// //     try {
// //         const prompt = getRecipeGenerationPrompt(ingredients, dietaryPreferences);
// //         const response = await openai.completions.create({
// //             model: 'gpt-2',  // Change to GPT-2
// //             prompt: prompt,
// //             max_tokens: 600,  // Adjust for GPT-2’s token limit
// //         });

// //         const _id = await saveOpenaiResponses({ userId, prompt, response })

// //         return { recipes: response.choices[0].text, openaiPromptId: _id || 'null-prompt-id' }
// //     } catch (error) {
// //         console.error('Failed to generate recipe:', error);
// //         throw new Error('Failed to generate recipe');
// //     }
// // };

// Function to call the OpenAI API to generate an image
const generateImage = (prompt: string): Promise<ImagesResponse> => {
    try {
        const response = openai.images.generate({
            model: 'dall-e-3',
            prompt,
            n: 1,
            size: '1024x1024',
        });

        // Return the response containing the image data
        return response;
    } catch (error) {
        throw new Error('Failed to generate image');
    }
};

export const generateImages = async (recipes: Recipe[], userId: string) => {
    try {
        const imagePromises: Promise<ImagesResponse>[] = recipes.map(recipe => generateImage(getImageGenerationPrompt(recipe.name, recipe.ingredients)));

        const images = await Promise.all(imagePromises);

        await saveOpenaiResponses({
            userId,
            prompt: `Image generation for recipe names ${recipes.map(r => r.name).join(' ,')} (note: not exact prompt)`,
            response: images
        })

        const imagesWithNames = images.map((imageResponse, idx) => (
            {
                imgLink: imageResponse.data[0].url,
                name: recipes[idx].name,
            }
        ));

        return imagesWithNames;
    } catch (error) {
        console.error('Error generating image:', error);
        throw new Error('Failed to generate image');
    }

};
// Define the response structure for Hugging Face API
interface HfValidationResponse {
    generated_text: string;
}

// Update the validateIngredient function
export interface ValidationResponse {
    isValid: boolean;
    possibleVariations: string[];
}

export interface ValidationResponse {
    isValid: boolean;
    possibleVariations: string[];
}

export const validateIngredient = async (ingredientName: string, userId: string): Promise<ValidationResponse | null> => {
    try {
        // Use the getIngredientValidationPrompt function to get the prompt
        const prompt = getIngredientValidationPrompt(ingredientName);
        
        // Make a request to Hugging Face's GPT-Neo model
        const response: HfValidationResponse = await hf.textGeneration({
            model: 'EleutherAI/gpt-neo-2.7B',
            inputs: prompt,
            parameters: {
                max_length: 200,
                do_sample: false,
                top_k: 50,
                top_p: 0.95,
                num_return_sequences: 1,
            },
        });

        console.log('Raw response from Hugging Face:', response);

        const validationText = response.generated_text;
        if (validationText) {
            console.log('Validation Text:', validationText); // Log the response
            try {
                // Trim whitespace from the output before parsing
                const trimmedText = validationText.trim();
                const jsonResponse: ValidationResponse = JSON.parse(trimmedText);
                await saveOpenaiResponses({ userId, prompt, response });
                return jsonResponse; 
            } catch (jsonError) {
                console.error('Failed to parse JSON response:', jsonError);
                return null; 
            }
        } else {
            console.error('No response text received from Hugging Face');
            return null;
        }
    } catch (error) {
        console.error('Failed to validate ingredient:', error);
        throw new Error('Failed to validate ingredient');
    }
};


// export const validateIngredient = async (ingredientName: string, userId: string): Promise<string | null> => {
//     try {
//         const prompt = getIngredientValidationPrompt(ingredientName);
//         const response = await openai.completions.create({
//             model: 'gpt-2',  // Change to GPT-2
//             prompt: prompt,
//             max_tokens: 400,  // Adjust for GPT-2’s token limit
//         });

//         await saveOpenaiResponses({ userId, prompt, response })

//         return response.choices[0].text
//     } catch (error) {
//         console.error('Failed to validate ingredient:', error);
//         throw new Error('Failed to validate ingredient');
//     }
// };
