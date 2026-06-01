# RAG Qualitative Comparison Report

> **Purpose:** For each of the 7 supported crops, a targeted disease question is asked twice — once through the full RAG-augmented pipeline and once with RAG forcibly disabled. The responses and retrieved source chunks are displayed side-by-side to assess the factual improvement that verified agronomic reference material provides.

| Field | Value |
|-------|-------|
| Test account | `rag_test@nava.local` |
| Field | RAG Test Field (ID 1) |
| LLM (chat) | Llama-3 70B via HF Router |
| LLM (router / keywords) | Llama-3.1-8B-Instruct |
| Embeddings | BAAI/bge-small-en-v1.5 (384-dim, local) |
| Retrieval | 5 semantic + ~5 keyword-filtered → rerank → top 3 |

---

## 1. Banana — Sigatoka (Black Sigatoka)

**Question asked:**

> My banana plants show dark streaks and yellow halos on the leaves — I think it is Black Sigatoka. What are the symptoms, cause, and recommended fungicide management for Black Sigatoka in banana?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on your description, it's likely that your banana plants are indeed infected with Black Sigatoka. Here's what you need to know:
>
> **Symptoms:** Black Sigatoka causes dark streaks or lesions on the leaves, often with yellow halos. These lesions can be faint and rusty brown in color, especially in the early stages. As the infection progresses, the lesions become darker, sink into the leaf, and turn into depressions. This can lead to the decline of plant function, and eventually, the death of the plant.
>
> **Cause:** Black Sigatoka is caused by the fungus Mycosphaerella fijiensis. It thrives in hot and humid tropical climates, which is ideal for banana cultivation. The fungus infects mature banana leaves and can spread rapidly in commercial areas where bananas are farmed in monoculture.
>
> **Recommended management:** To manage Black Sigatoka, I recommend the following:
>
> 1. **Remove infected leaves:** Remove leaves with mature spots to prevent the spread of the disease.
> 2. **Improve air circulation:** Increase plant spacing to improve air circulation and reduce humidity, making it less conducive to the fungus.
> 3. **Fungicide applications:** Regular fungicide applications may be necessary, especially in export plantations. However, please consult local guidelines for the recommended types and frequencies of fungicide applications, as the pathogen has developed resistance to some fungicides.
>
> Remember to also practice good sanitation and hygiene to prevent the spread of the disease. If you're unsure about the best course of action or need further guidance, consider consulting with a local agronomist or crop expert.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | PlantVillage | Black sigatoka (Black leaf streak) (Mycosphaerella fijiensis) - Management | [Black sigatoka (Black leaf streak) (Mycosphaerella fijiensis) - Management] Export plantations may require regular fungicide applications; increase plant spacing to improve air circulation and reduce humidity; remove leaves with mature spots |
| 2 | Wikipedia | Black sigatoka - Importance | [Black sigatoka - Importance] The worldwide spread of the disease has been rapid, with its naming and first reported occurrence in 1963.[18][19] The disease was reported in 1972 in Honduras, from where it spread north to central Mexico and south to Brazil and into the Caribbean islands[19] in 1991.[ |
| 3 | Wikipedia | Black sigatoka - Symptoms | [Black sigatoka - Symptoms] Black Sigatoka is also known as black leaf streak, causing streaks that run parallel to the secondary veins of banana leaves. It affects banana trees specifically in tropical climates; including Asia, East and West Africa, Trinidad, Grenada, many Pacific islands, Central… |

### ❌ Answer WITHOUT RAG

> I'm happy to help you with that!
>
> Symptoms of Black Sigatoka (Mycosphaerella fijiensis) on banana plants include:
>
> * Dark streaks or blotches on the leaves, often with a yellow halo
> * Premature yellowing and death of leaves, starting from the oldest leaves
> * Reduced fruit production and quality
>
> The cause of Black Sigatoka is a fungal infection, which thrives in warm, humid environments like the one you have in Kottayam, Kerala during the monsoon season.
>
> For fungicide management, I recommend a combination of cultural practices and targeted fungicide applications. Here's a suggested approach:
>
> 1. Cultural practices:
> 	* Remove and dispose of infected leaves to prevent the spread of the disease.
> 	* Improve air circulation within the plantation by pruning nearby plants.
> 	* Avoid excessive nitrogen fertilization, which can exacerbate the disease.
> 2. Fungicide applications:
> 	* Apply a systemic fungicide containing triazoles (e.g., propiconazole or tebuconazole) or strobilurins (e.g., azoxystrobin) at the recommended label rates.
> 	* Spray the fungicide every 14-21 days, depending on the product and weather conditions.
> 	* Make sure to cover both sides of the leaves and the soil around the plants.
>
> Please consult local guidelines or an agronomist for specific fungicide recommendations and dosage rates, as these may vary depending on your region and regulations.
>
> Remember to monitor your plants closely and adjust your management strategy accordingly. If you have any further questions or concerns, feel free to ask!

---

## 2. Cassava — Cassava Mosaic Disease

**Question asked:**

> My cassava plants have distorted, mosaic-patterned leaves with yellowing. What causes Cassava Mosaic Disease, how does it spread, and what are the management strategies to control it?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on your description, it's likely that your cassava plants are infected with Cassava Mosaic Disease (CMD) caused by the African cassava mosaic virus (ACMV). 
>
> CMD is spread by infected cuttings and by whiteflies. To manage the disease, it's essential to use only healthy and disease-free cuttings for planting. You should also consider planting cassava varieties that are more tolerant to the virus. 
>
> Roguing and sanitation are crucial: remove and destroy any plants showing symptoms of the disease, including alternative hosts. Early harvesting of tubers can help minimize losses due to necrosis. 
>
> To control the whitefly vector, encourage beneficial insects in the field, such as spiders and ladybird beetles. You can also use yellow sticky traps to monitor whitefly infestation and spray insecticidal soaps under the leaf surface to kill the flies.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | PlantVillage | Cassava Mosaic Disease (African cassava mosaic virus ACMV) - Symptoms | [Cassava Mosaic Disease (African cassava mosaic virus ACMV) - Symptoms] Discolored pale green, yellow or white mottled leaves which may be distorted with a reduced size; in highly susceptible cassava cultivars plant growth may be stunted, resulting in poor root yield and low quality stem cuttings. N |
| 2 | PlantVillage | Cassava Mosaic Disease (African cassava mosaic virus ACMV) - Comments | [Cassava Mosaic Disease (African cassava mosaic virus ACMV) - Comments] Disease is spread by infected cuttings and by whiteflies. The leaves are yellow, mottled and distorted. If leaves are yellow all over but are a normal size or there brown leaves that that does not indicate disease. Wild cassava… |
| 3 | PlantVillage | Cassava Brown Streak Disease (Cassava brown streak virus CBSV) - Management | [Cassava Brown Streak Disease (Cassava brown streak virus CBSV) - Management] Disease diagnosis: The first and foremost important aspect is to identify the disease correctly. Cassava brown streak disease varies in symptoms which made it difficult to identify in the field. It makes further complicate |

### ❌ Answer WITHOUT RAG

> Cassava Mosaic Disease (CMD) is a major concern! It's a viral disease caused by the Cassava Mosaic Virus (CMV), which is transmitted by the whitefly vector (Bemisia tabaci). The symptoms you've described, such as distorted, mosaic-patterned leaves with yellowing, are classic signs of CMD.
>
> The virus is spread through:
>
> 1. Whitefly vectors: Infected whiteflies can transmit the virus to healthy cassava plants while feeding.
> 2. Infected planting material: Using infected stem cuttings or seedlings can spread the disease.
> 3. Contact between plants: Physical contact between infected and healthy plants can also transmit the virus.
>
> To manage and control CMD, I recommend the following strategies:
>
> 1. **Use certified disease-free planting material**: Ensure that your stem cuttings or seedlings are obtained from a trusted source and are free from the virus.
> 2. **Implement whitefly management**: Control whitefly populations through integrated pest management (IPM) techniques, such as introducing natural predators, using sticky traps, or applying insecticides as a last resort.
> 3. **Remove and destroy infected plants**: Remove and destroy any infected plants to prevent the spread of the disease to healthy plants.
> 4. **Practice good sanitation**: Regularly clean and disinfect tools and equipment to prevent the spread of the virus.
> 5. **Monitor your field regularly**: Keep a close eye on your cassava plants and remove any infected plants as soon as possible.
> 6. **Consider resistant varieties**: If possible, plant cassava varieties that have built-in resistance to CMD.
>
> Remember, it's essential to act quickly to prevent the spread of CMD. If you're unsure about the diagnosis or need further guidance, consider consulting with a local agronomist or expert.

---

## 3. Corn — Northern Leaf Blight

**Question asked:**

> I see long, greyish-tan lesions running along the corn leaves. Can you explain Northern Leaf Blight in corn — what fungus causes it, what conditions favour it, and what fungicide or cultural practices are recommended?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on your description of the lesions, it's likely that your corn crop is affected by Northern Leaf Blight (NLB). NLB is caused by the fungus Exserohilum turcicum.
>
> Northern Leaf Blight thrives in conditions with high humidity and moderate temperatures, which is consistent with the current weather conditions in your field (94% humidity and 25°C temperature). The disease can be easily identified in the field due to its long, narrow lesions that are unrestricted by veins.
>
> To manage NLB, I recommend the following:
>
> 1. Crop rotation: Rotate your corn crop with a non-host crop to reduce the fungus inoculum from crop debris.
> 2. Resistant varieties: Grow available resistant varieties of corn to reduce the risk of NLB.
> 3. Proper tillage: Follow proper tillage practices to reduce the fungus inoculum from crop debris.
>
> In severe cases of disease incidence, applying a suitable fungicide may be necessary. However, I would advise consulting local guidelines or an agronomist for specific recommendations on fungicide application rates and timings.
>
> Remember to monitor your crop regularly for any signs of disease, and take prompt action if you notice any further symptoms.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | PlantVillage | Northern Leaf Blight (Exserohilum turcicum) - Management | [Northern Leaf Blight (Exserohilum turcicum) - Management] Follow proper tillage to reduce fungus inoculum from crop debris. Follow crop rotation with non host crop. Grow available resistant varieties. In severe case of disease incidence apply suitable fungicide. |
| 2 | PlantVillage | Northern Leaf Blight (Exserohilum turcicum) - Symptoms | [Northern Leaf Blight (Exserohilum turcicum) - Symptoms] In the beginning we will notice elliptical gray-green lesions on leaves. As the disease process this lesions become pale gray to tan color. Later stage the lesions looks dirty due to dark gray spores particularly under lower leaf surface. The… |
| 3 | Wikipedia | Common Corn Rust​ - Overview | [Common Corn Rust​ - Overview] Common rust is a disease that occurs in temperate to subtropical areas, including the Midwest. Though it cannot overwinter in the Midwest, windborne spores reach northern areas of the US and Canada from southern areas where overwintering is possible. Common rust is rar |

### ❌ Answer WITHOUT RAG

> The lesions you're describing are a classic symptom of Northern Leaf Blight (NLB) in corn! NLB is caused by the fungus Setosphaeria turcica. 
>
> This disease thrives in cool, humid environments, which is consistent with the current weather conditions in your field (high humidity and moderate temperature). Prolonged leaf wetness, especially during the vegetative stage, can exacerbate the issue.
>
> To manage NLB, I recommend a combination of cultural practices and fungicide applications. For cultural practices:
>
> 1. Improve air circulation: Ensure proper plant spacing and remove any weeds to promote air movement, reducing humidity around the plants.
> 2. Remove infected leaves: Remove and dispose of infected leaves to prevent the fungus from spreading.
> 3. Irrigation management: Avoid overhead irrigation, which can exacerbate the issue. Instead, use drip irrigation or soaker hoses to reduce leaf wetness.
>
> For fungicide applications, I suggest consulting local guidelines or an agronomist for specific recommendations on the type and dosage of fungicides suitable for your region. However, strobilurin and triazole-based fungicides are commonly used to control NLB.
>
> Keep in mind that it's essential to monitor your field regularly for any signs of disease and take prompt action to prevent the spread of NLB.

---

## 4. Cucumber — Powdery Mildew

**Question asked:**

> My cucumber plants have white powdery patches on the upper surface of the leaves. What is Powdery Mildew in cucumber, what causes it, and what are the treatment and prevention measures?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Powdery mildew is a common fungal disease affecting cucumbers, characterized by white, powdery patches on the upper surface of leaves. The causal fungi are Podosphaera xanthii and Erysiphe cichoracearum. 
>
> The disease develops quickly under favorable conditions, which include dense plant growth, low light intensity, high relative humidity, and temperatures between 68-80°F (20-27°C). 
>
> Symptoms include white, powdery fungal growth on the upper surface of leaves, stems, and fruits. As the disease progresses, infected leaves become yellow, distorted, and may drop prematurely.
>
> To manage powdery mildew, a combination of cultural, biological, and chemical controls is recommended. Here are some measures:
>
> **Cultural Controls:**
>
> 1. **Plant resistant varieties**: Many commercial cucumber cultivars have built-in resistance to powdery mildew.
> 2. **Improve air circulation**: Increase row and plant spacing to reduce humidity and promote air movement.
> 3. **Water carefully**: Avoid overhead watering, and instead use drip irrigation or soaker hoses to reduce leaf wetness.
> 4. **Remove infected leaves**: Dispose of infected leaves to prevent the spread of the disease.
>
> **Biological Controls:**
>
> 1. **Fungicides containing antagonistic fungi**: Apply biological control products containing fungi that antagonize the powdery mildew pathogen.
>
> **Chemical Controls:**
>
> 1. **Fungicides**: Apply fungicides containing mobile or systemic ingredients, such as Vivando, Proline, or Procure, every 7-10 days starting at the onset of disease development. Always follow the recommended label rates and resistance management strategies to minimize the risk of fungicide resistance.
>
> **Prevention Measures:**
>
> 1. **Monitor your plants regularly**: Inspect your plants weekly, especially during fruit formation, to detect powdery mildew early.
> 2. **Maintain good garden hygiene**: Remove weeds, debris, and infected plant material to reduce the risk of disease spread.
> 3. **Use crop rotation**: Rotate cucumber crops with non-cucurbit crops to break the disease cycle.
>
> Remember to integrate these measures into your disease management strategy to effectively control powdery mildew in your cucumber plants.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | Cornell University | Cucurbit Powdery Mildew - Overview | [Cucurbit Powdery Mildew - Overview] Impact and causal fungi Powdery mildew is a common disease of cucurbits under field and greenhouse conditions in most areas of the world. Although all cucurbits are susceptible, symptoms are less common on cucumber and melon because many commercial cultivars have |
| 2 | Nova Scotia Vegetable | Downy Mildew: - Overview | [Downy Mildew: - Overview] The signs of downy mildew include lesions on the leaves (Figure 1). Angular, vein-delimited lesions that may initially appear slightly chlorotic are typical disease symptoms on cucumbers, and pathogen sporulation is easily visible on the lower leaf surface. Under humid and |
| 3 | PlantVillage | Powdery mildew (Erysiphe cichoracearum / Sphaerotheca fuliginea) - Symptoms | [Powdery mildew (Erysiphe cichoracearum / Sphaerotheca fuliginea) - Symptoms] The appearance of white powdery spots on the upper surfaces of leaves, stems and fruits. As the disease progress, white fungal growth covers whole leaves and stem. The infected leaves become yellow, distorted and may drop… |

### ❌ Answer WITHOUT RAG

> Powdery Mildew is a common fungal disease affecting cucumbers, especially in humid and warm conditions like those in Kottayam, Kerala during the Monsoon Season. The white powdery patches on the upper surface of the leaves are a clear indication of this disease.
>
> Causes:
> Powdery Mildew is caused by the fungus Podosphaera xanthii. It thrives in humid environments with temperatures between 15°C to 25°C, which is exactly the current condition in your field. High humidity, overhead watering, and poor air circulation can contribute to the development of this disease.
>
> Treatment:
> For mild infections, remove and dispose of infected leaves to prevent the disease from spreading. For more severe cases, consider using fungicides containing copper, sulfur, or neem oil. However, please consult local guidelines or an agronomist for specific dosage recommendations.
>
> Prevention Measures:
> 1. Improve air circulation: Ensure adequate spacing between plants and rows to promote good air movement.
> 2. Water management: Avoid overhead watering, and instead, use drip irrigation or soaker hoses to reduce moisture on the leaves.
> 3. Crop monitoring: Regularly inspect your plants for signs of Powdery Mildew and take action promptly.
> 4. Resistant varieties: Consider growing cucumber varieties with built-in resistance to Powdery Mildew.
> 5. Sanitation: Remove weeds and debris from the field, as they can harbor the fungus.
>
> Remember to prioritize disease detection results and take necessary actions to prevent the spread of Powdery Mildew in your cucumber crop.

---

## 5. Rice — Rice Blast

**Question asked:**

> Diamond-shaped lesions with grey centres and brown borders are appearing on my rice leaves. What is Rice Blast, what pathogen causes it, and what are the recommended chemical and cultural management practices?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on the symptoms you've described, it's likely that your rice crop is infected with Rice Blast, a fungal disease caused by Magnaporthe grisea. 
>
> Rice Blast is a highly destructive disease that can cause significant yield losses, especially in areas with intense rice cultivation. The pathogen thrives in environments with high soil nitrogen content.
>
> To manage Rice Blast, I recommend a combination of chemical and cultural practices:
>
> Chemical control:
> Consult local guidelines for recommended fungicides and application rates. It's essential to note that chemical control should be used judiciously and in conjunction with cultural practices to avoid the development of fungicide-resistant pathogen populations.
>
> Cultural control:
> 1. Crop rotation: Rotate rice with non-cereal crops to break the disease cycle.
> 2. Sanitation: Remove and dispose of infected plant debris to prevent the spread of the disease.
> 3. Water management: Avoid excessive irrigation, as it can exacerbate the disease. Drain fields regularly to reduce moisture levels.
> 4. Nutrient management: Optimize nitrogen fertilizer application to avoid excessive soil nitrogen, which can contribute to disease development.
> 5. Resistant varieties: Consider planting rice varieties with built-in resistance to Rice Blast.
>
> Remember to monitor your crop regularly for disease symptoms and adjust your management strategy accordingly.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | PlantVillage | Rice blast (Magnaporthe grisea) - Symptoms | [Rice blast (Magnaporthe grisea) - Symptoms] Lesions on all parts of shoot; white to green or gray diamond-shaped lesions with dark green borders; death of leaf blades; black necrotic patches on culm; rotting panicles |
| 2 | PlantVillage | Rice blast (Magnaporthe grisea) - Comments | [Rice blast (Magnaporthe grisea) - Comments] Most important disease of rice worldwide; causes most damage in areas of intense cultivation; disease emergence favors high soil nitrogen content |
| 3 | PlantVillage | Sheath blight (Rhizoctonia solani) - Comments | [Sheath blight (Rhizoctonia solani) - Comments] Occurs in all areas where rice is grown; second only importance to rice blast; most damaging in intensive rice production; spreads rapidly via irrigation |

### ❌ Answer WITHOUT RAG

> Rice Blast is a serious fungal disease affecting rice crops, and it seems like your field is experiencing an outbreak. Rice Blast is caused by the fungus Magnaporthe oryzae.
>
> For chemical management, I recommend applying a fungicide specifically designed to combat Rice Blast, such as tricyclazole or azoxystrobin. However, please consult local guidelines for the recommended dosage and application timing.
>
> In addition to chemical control, cultural management practices can help mitigate the spread of the disease. Here are some recommendations:
>
> 1. Remove infected leaves or plants to prevent the fungus from spreading.
> 2. Improve air circulation within the field by adjusting plant spacing or pruning nearby plants.
> 3. Avoid excessive nitrogen fertilization, as it can promote the growth of the fungus.
> 4. Water management is crucial; avoid flooding the field, and ensure good drainage to reduce humidity.
> 5. Consider using resistant rice varieties in future plantings.
>
> It's essential to take prompt action to control the spread of Rice Blast. Monitor your field closely, and consider consulting with a local agronomist for personalized guidance.

---

## 6. Soybean — Soybean Downy Mildew

**Question asked:**

> I can see pale green to yellow patches on the upper surface of soybean leaves with grey-purple fuzz underneath. What is Soybean Downy Mildew, what causes it, and how should I manage it?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on your description, it's likely that your soybean crop is infected with Downy Mildew, a common fungal leaf disease caused by Peronospora manshurica. The pale green to yellow patches on the upper surface of the leaves, accompanied by grey-purple fuzz underneath, are characteristic symptoms of this disease.
>
> Downy Mildew thrives in cool, wet, or humid weather conditions, which is consistent with the current weather in your field (high humidity and recent precipitation). The pathogen survives in crop residue and on seed surfaces, and infection can spread quickly through a field during periods of cool, wet, or humid weather.
>
> To manage Downy Mildew, I recommend the following:
>
> 1. Treat seeds with an appropriate fungicide prior to planting to prevent the disease from spreading.
> 2. Plow soybean residue into the soil after harvest to reduce the amount of infected material that can serve as a source of infection for future crops.
> 3. Consider growing soybean varieties that are resistant to Downy Mildew, if possible.
> 4. Rotate your crop away from soybean for a one-year period to break the disease cycle.
>
> It's essential to take prompt action to control the spread of Downy Mildew, as it can significantly impact soybean yields and quality.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | Crop Protection Network | Downy Mildew - Overview | [Downy Mildew - Overview] It is a common fungal leaf disease of soybean that occurs wherever soybeans are grown, especially when weather conditions are rainy and humid. The downy mildew pathogen, Peronospora manshurica, survives in crop residue and on the seed surface. The pathogen is in the same gr |
| 2 | PlantVillage | Phytophthora rot (Phytophthora megasperma) - Symptoms | [Phytophthora rot (Phytophthora megasperma) - Symptoms] Susceptible varieties of soybean may have water-soaked stems and yellowing leaves; yellowing occurs between leaf veins and along leaf margins; upper leaves of plant become chlorotic and wilted; tolerant soybean varieties may show stunted growth |
| 3 | PlantVillage | Downy mildew (Peronospora manshurica) - Management | [Downy mildew (Peronospora manshurica) - Management] Treat seeds with an appropriate fungicide prior to planting; plow soybean residue into soil after harvest; grow soybean varieties which are resistant to the disease where possible; rotate crop away from soybean for a one year period |

### ❌ Answer WITHOUT RAG

> Those symptoms are quite distinctive! Based on your description, I'm confident that your soybean crop is suffering from Soybean Downy Mildew (SDM). 
>
> Soybean Downy Mildew is a fungal disease caused by Peronospora manshurica. It thrives in humid and cool environments, which is consistent with the current weather conditions in your field (high humidity and moderate temperature). The grey-purple fuzz you observed is the sporulation of the fungus on the underside of the leaves.
>
> To manage SDM, I recommend the following:
>
> 1. Remove and dispose of infected leaves to prevent the fungus from spreading.
> 2. Improve air circulation within the field by adjusting plant spacing or pruning nearby plants.
> 3. Apply a fungicide specifically designed to combat downy mildew. However, please consult local guidelines or an agronomist for the recommended dosage and application schedule.
> 4. Consider using resistant soybean varieties in future plantings to reduce the risk of SDM.
>
> It's essential to act promptly, as SDM can significantly impact soybean yields and quality. Keep a close eye on your crop, and let me know if you have any further questions or concerns!

---

## 7. Tomato — Tomato Late Blight

**Question asked:**

> Dark, water-soaked lesions are appearing on my tomato leaves and spreading to the stems and fruits. What is Tomato Late Blight, what organism causes it, and what are the recommended management and fungicide options?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on your description, it's likely that your tomato plants are infected with Late Blight (Phytophthora infestans). This disease affects all aerial parts of the tomato plant, causing water-soaked green to black areas on leaves, which rapidly change to brown lesions. The causal organism is Phytophthora infestans (Mont.) de Bary.
>
> To manage Late Blight, I recommend the following:
>
> 1. Remove and dispose of infected leaves and stems to prevent the disease from spreading.
> 2. Improve air circulation around the plants by staking or caging them to raise them off the ground.
> 3. Mulch around the base of the plants to reduce soil splashing, which may contain fungal spores.
> 4. Avoid overhead watering, and instead use a soaker hose at the base of the plant to keep the foliage dry.
> 5. Control weeds, especially those in the nightshade family, which can host the disease.
>
> As for fungicide options, you can consider using chlorothalonil, mancozeb, or copper-based fungicides like copper hydroxide, copper sulfate, or copper oxychloride sulfate. However, please follow the recommended application rates, intervals, and harvest restrictions listed on the pesticide label.
>
> It's essential to note that fungicides will not cure infected leaves, but they can protect new leaves from becoming infected. Apply fungicides at 7 to 10-day intervals throughout the season.
>
> Remember to consult local guidelines or an agronomist for specific advice on fungicide usage and dosage in your area.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | PlantVillage | Late blight (Phytophthora infestans) - Symptoms | [Late blight (Phytophthora infestans) - Symptoms] Late blight affects all aerial parts of the tomato plant; initial symptoms of the disease appear as water-soaked green to black areas on leaves which rapidly change to brown lesions; fluffy white fungal growth may appear on infected areas and leaf un |
| 2 | Tamilnadu Agricultural University | Late blight  Phytophthora infestans - Symptoms | [Late blight  Phytophthora infestans - Symptoms] Water-soaked black lesions on leaves and stems Lesions expand rapidly and the entire leaf becomes necrotic. White sporulation (sporangia and sporangiophores) on leaves. Dark brown lesions on fruit Soft rot and disintegration of fruits.  Causal organis |
| 3 | Missouri Botanical Garden | Septoria leaf spot is caused by a fungus, Septoria lycopersici. It is one of the most destructive diseases of tomato foliage and is particularly severe in areas where wet, humid weather persists for extended periods. - Overview | [Septoria leaf spot is caused by a fungus, Septoria lycopersici. It is one of the most destructive diseases of tomato foliage and is particularly severe in areas where wet, humid weather persists for extended periods. - Overview] Symptoms and Diagnosis Septoria leaf spot usually appears on the lower |

### ❌ Answer WITHOUT RAG

> Tomato Late Blight is a severe and destructive disease caused by the oomycete pathogen Phytophthora infestans. It's a major concern for tomato growers, especially in humid and cool weather conditions like we're experiencing in Kottayam, Kerala during the Monsoon Season.
>
> For management, I recommend the following:
>
> 1. Remove and dispose of infected plants to prevent the disease from spreading.
> 2. Improve air circulation by providing sufficient spacing between plants and removing weeds.
> 3. Avoid overhead irrigation, which can exacerbate the disease. Instead, use drip irrigation or soaker hoses.
> 4. Apply organic amendments like copper-based fungicides or resistant starch to help control the disease.
>
> As for fungicide options, I suggest consulting local guidelines for specific recommendations. However, some commonly used fungicides for managing Tomato Late Blight include copper oxychloride, chlorothalonil, and azoxystrobin. It's essential to rotate fungicides with different modes of action to minimize the risk of resistance development.
>
> Remember to always follow the recommended dosage and application instructions for any fungicide you choose to use.

---

## Summary

> The side-by-side comparison above shows how Yukthi (RAG) grounds each answer in verified agronomic source material — reducing hallucinated dosages, uncertain hedging, and missing specifics compared to the parametric-only baseline.
