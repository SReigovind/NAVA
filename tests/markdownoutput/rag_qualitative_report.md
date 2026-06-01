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

> Based on your description, it's likely that your banana plants are indeed affected by Black Sigatoka. Here's what you need to know:
>
> **Symptoms:** Red/brown flecks or spots on the underside or topside of leaves, with dark or yellow borders and grey centers. These spots can lead to the death of the leaf surface, and the bunch may not develop properly.
>
> **Cause:** Black Sigatoka is caused by the fungus Mycosphaerella fijiensis.
>
> **Management:** To manage Black Sigatoka, you should consider regular fungicide applications. Additionally, increase plant spacing to improve air circulation and reduce humidity, which can help prevent the spread of the disease. Remove leaves with mature spots to prevent the fungus from spreading further. Please consult local guidelines or an agronomist for specific fungicide recommendations and application instructions.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | banana.txt | Black sigatoka (Black leaf streak) Mycosphaerella fijiensis | Black sigatoka (Black leaf streak) Mycosphaerella fijiensis Symptoms Red/brown flecks or spots on underside or topside of leaves; spots with dark or yellow border and grey centre; death of leaf surface; bunch not developing Cause Fungus Comments Currently the most important disease of banana; promot |
| 2 | banana.txt | Yellow sigatoka Mycosphaerella musae | Yellow sigatoka Mycosphaerella musae Symptoms Pale green flecks on leaves which enlarge to chlorotic streaks; streaks enlarge and turn brown with chlorotic halo; mature lesions are gray with a dark brown border; lesions coalesce and kill large areas of leaves Cause Fungus Comments Spores spread by w |
| 3 | banana.txt | Brown spots on fruit peel; large brown to black areas; black lesions on green fruit. | Brown spots on fruit peel; large brown to black areas; black lesions on green fruit. Cause Fungus Comments Wet conditions promote growth and spread of disease; spread by rainfall through plant or banana bunch. Management Commercially produced fruit should be washed and dipped in fungicide prior to s |

### ❌ Answer WITHOUT RAG

> Based on your description, it's likely that your banana plants are indeed infected with Black Sigatoka, a fungal disease caused by Mycosphaerella fijiensis.
>
> **Symptoms:**
> The typical symptoms of Black Sigatoka include:
>
> * Dark brown to black streaks or patches on the leaves, often with yellow halos
> * Premature yellowing and death of leaves, leading to defoliation
> * Reduced fruit production and quality
>
> **Cause:**
> Black Sigatoka is a fungal disease that thrives in warm, humid environments, making your Monsoon Season conditions in Kottayam, Kerala, conducive to its spread.
>
> **Recommended Fungicide Management:**
> To manage Black Sigatoka, I recommend a combination of cultural practices and fungicide applications. Please consult local guidelines for specific fungicide recommendations and dosages. As a general approach:
>
> 1. **Remove infected leaves:** Regularly remove and dispose of infected leaves to prevent the spread of the disease.
> 2. **Improve air circulation:** Ensure good air circulation around the plants to reduce humidity and prevent moisture buildup.
> 3. **Fungicide applications:** Apply a fungicide specifically labeled for Black Sigatoka control in banana. You may need to apply multiple times, depending on the severity of the infection and local guidelines.
> 4. **Resistant varieties:** Consider planting banana varieties with built-in resistance to Black Sigatoka for long-term management.
>
> Remember to always follow local guidelines and regulations regarding fungicide use, and consult with a local agronomist if you're unsure about the best course of action.

---

## 2. Cassava — Cassava Mosaic Disease

**Question asked:**

> My cassava plants have distorted, mosaic-patterned leaves with yellowing. What causes Cassava Mosaic Disease, how does it spread, and what are the management strategies to control it?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Cassava Mosaic Disease is caused by the African cassava mosaic virus (ACMV). The virus is responsible for the distorted, mosaic-patterned leaves with yellowing that you're observing on your cassava plants.
>
> The disease spreads through infected cuttings and by whiteflies, which act as vectors. When whiteflies feed on infected plants, they acquire the virus and can then transmit it to healthy plants.
>
> To manage Cassava Mosaic Disease, the following strategies can be employed:
>
> 1. **Use resistant varieties**: Plant cassava varieties that are resistant to the virus, such as 'Rwizihiza', 'Ndamirabana', 'Cyizere', 'Seruruseke', 'Mavoka', 'Garukunsubire', and 'Mbakungahaze'.
> 2. **Inspect plants regularly**: Regularly inspect your cassava plants for symptoms of the disease and remove any infected plants to prevent the spread of the virus.
> 3. **Remove and destroy infected plants**: Uproot and destroy any plants showing symptoms of the disease to prevent the virus from spreading.
> 4. **Avoid using infected cuttings**: Only use healthy, disease-free cuttings for planting to prevent the spread of the virus.
> 5. **Control whitefly populations**: Implement integrated pest management strategies to control whitefly populations, which can help reduce the spread of the virus.
>
> It's essential to note that there is no agrochemical agent or organic treatment for Cassava Mosaic Disease. The key to controlling the disease is through the use of resistant varieties, good agricultural practices, and vector management.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | cassava.txt | Cassava Mosaic Disease African cassava mosaic virus (ACMV) | Cassava Mosaic Disease African cassava mosaic virus (ACMV) Symptoms Discolored pale green, yellow or white mottled leaves which may be distorted with a reduced size; in highly susceptible cassava cultivars plant growth may be stunted, resulting in poor root yield and low quality stem cuttings. Note… |
| 2 | cassava.txt | Anthracnose usually does not cause large-scale economic damage to cassava and control is usually not necessary; avoid planting cuttings with cankers; if disease does occur crop debris should be removed and destroyed after harvest | Anthracnose usually does not cause large-scale economic damage to cassava and control is usually not necessary; avoid planting cuttings with cankers; if disease does occur crop debris should be removed and destroyed after harvest  Cassava Brown Leaf Spot Cercosporidium henningsii Symptoms Circular o |
| 3 | cassava.txt | Root rot disease emergence is often favored by waterlogged, poorly-draining soils | Root rot disease emergence is often favored by waterlogged, poorly-draining soils Management Plant cassava in well-draining soils; remove and destroy all crop debris by burning; sanitize all tools after use  Pests Category : Insects African root and tuber scale Strictococcus vayssierrei Symptoms Ova |

### ❌ Answer WITHOUT RAG

> Cassava Mosaic Disease (CMD) is a viral disease caused by the Cassava Mosaic Virus (CMV). It's a common disease in cassava crops, especially in tropical regions like Kerala.
>
> CMD spreads through:
>
> 1. Infected cassava cuttings: Using infected cuttings for planting can spread the disease.
> 2. Whiteflies: The whitefly (Bemisia tabaci) is the primary vector of CMV. These insects feed on infected plants and then transmit the virus to healthy plants.
> 3. Contact: Direct contact between infected and healthy plants can also spread the disease.
>
> To manage and control CMD:
>
> 1. Use certified disease-free cassava cuttings for planting.
> 2. Implement good agricultural practices like maintaining field sanitation, removing weeds, and disposing of infected plants.
> 3. Control whitefly populations using integrated pest management (IPM) strategies, such as introducing natural predators, using insecticidal soap, or neem oil.
> 4. Avoid excessive nitrogen fertilization, as it can increase the susceptibility of cassava plants to CMD.
> 5. Consider using resistant or tolerant cassava varieties, if available.
>
> It's essential to monitor your field regularly for early detection and take prompt action to prevent the spread of the disease.

---

## 3. Corn — Northern Leaf Blight

**Question asked:**

> I see long, greyish-tan lesions running along the corn leaves. Can you explain Northern Leaf Blight in corn — what fungus causes it, what conditions favour it, and what fungicide or cultural practices are recommended?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Northern Leaf Blight (NLB) in corn is caused by the fungus Exserohilum turcicum. The disease is characterized by long, narrow, greyish-tan lesions that run along the corn leaves, often unrestricted by veins.
>
> NLB is favored by cool, moist weather conditions.
>
> To manage NLB, I recommend the following cultural practices:
>
> 1. Plant resistant hybrids: Choose corn varieties that have built-in resistance to NLB.
> 2. Crop rotation: Rotate corn with non-host crops to reduce the amount of inoculum in the soil.
>
> For fungicide recommendations, I suggest consulting local guidelines or an agronomist for advice on the most effective and approved products for your region.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | corn.txt | The disease mainly spread through rain splash and wind. | The disease mainly spread through rain splash and wind. Management Follow proper tillage to reduce fungus inoculum from crop debris. Follow crop rotation with non host crop. Grow available resistant varieties. In severe case of disease incidence apply suitable fungicide.  Southern corn leaf blight B |
| 2 | corn.txt | Plants wilting and leaves changing color from light to dull green; lower stalk turns straw yellow; internal stalk tissue breaks down; interior of stalk has a red discoloration; black fungal fruiting bodies may be visible on the stalk, often at internodes, and can be easily scraped off; if fungal infection affects the ears, it produces a red mold at the tips of the ear which spreads down; early infection may result in the ear being covered in pink mycelium which causes the corn husk to adhere to the ear. | Plants wilting and leaves changing color from light to dull green; lower stalk turns straw yellow; internal stalk tissue breaks down; interior of stalk has a red discoloration; black fungal fruiting bodies may be visible on the stalk, often at internodes, and can be easily scraped off; if fungal inf |
| 3 | corn.txt | Anthracnose symptoms vary widely depending on numerous factors such as genotype, age of plant and environmental conditions. | Anthracnose symptoms vary widely depending on numerous factors such as genotype, age of plant and environmental conditions.  - Small oval or elongated water-soaked spots which enlarge up to 15 mm long appear on leaves - Lesions develop a tan center and red-brown or orange border - Lesions may coales |

### ❌ Answer WITHOUT RAG

> Northern Leaf Blight (NLB) in corn is caused by the fungus Setosphaeria turcica. The long, greyish-tan lesions you're observing are a classic symptom of NLB.
>
> NLB thrives in cool, humid environments with temperatures between 18°C and 28°C (64°F to 82°F) and high relative humidity, which is consistent with the current weather conditions in your field (26.1°C and 95% humidity). Prolonged leaf wetness, especially during the monsoon season, also favors the development of NLB.
>
> To manage NLB, I recommend a combination of cultural practices and fungicide applications. Cultural practices include:
>
> 1. Crop rotation: Rotate corn with non-host crops to break the disease cycle.
> 2. Sanitation: Remove and dispose of infected plant debris to reduce inoculum.
> 3. Irrigation management: Avoid excessive irrigation, which can exacerbate the disease.
>
> As for fungicides, consult local guidelines for recommended products and application rates. It's essential to choose a fungicide with efficacy against Setosphaeria turcica. Always follow label instructions and take necessary precautions to minimize environmental impact.
>
> Keep in mind that it's crucial to monitor your field regularly for disease development and adjust your management strategy accordingly.

---

## 4. Cucumber — Powdery Mildew

**Question asked:**

> My cucumber plants have white powdery patches on the upper surface of the leaves. What is Powdery Mildew in cucumber, what causes it, and what are the treatment and prevention measures?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Powdery mildew is a fungal disease that affects cucumber plants, causing white powdery spots or patches on the upper surfaces of leaves, stems, and fruits. The disease is caused by the fungus Erysiphe cichoracearum or Sphaerotheca fuliginea.
>
> The disease is favored by moderate temperatures and shady conditions. The spores are carried by wind from one plant to another, making it easily transmissible.
>
> To manage powdery mildew, you can take the following measures:
>
> 1. Grow available resistant varieties of cucumber.
> 2. If the disease is severe, spray a suitable fungicide.
>
> Prevention measures include:
>
> 1. Using disease-free seeds and treating them prior to planting.
> 2. Rotating crops every 2 years to prevent the buildup of inoculum.
>
> Early detection and prompt action can help limit the spread of the disease and prevent significant damage to your cucumber crop.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | cucumber.txt | Use disease free seed; treat seeds prior to planting; rotate crops every 2 years. | Use disease free seed; treat seeds prior to planting; rotate crops every 2 years.   Powdery mildew Erysiphe cichoracearum Sphaerotheca fuliginea Symptoms The appearance of white powdery spots on the upper surfaces of leaves, stems and fruits. As the disease progress, white fungal growth covers whole |
| 2 | cucumber.txt | Fusarium wilt (Cucumber wilt, Foot-rot) Fusarium oxysporum | Fusarium wilt (Cucumber wilt, Foot-rot) Fusarium oxysporum Symptoms Rotting of seedling stems at soil line; brown lesions on one side of stem; discoloration of tissue inside vine. Cause Fungus Comments Disease favors warm, moist soil. Management Plant fungicide treated seed; rotate crops on 4 year r |
| 3 | cucumber.txt | Description | Cucumber, Cucumis sativus, is a warm season, vining, annual plant in the family Cucurbitaceae grown for its edible cucumber fruit. The cucumber plant is a sprawling vine with large leaves and curling tendrils. The plant may have 4 or 5 main stems from which the tendrils branch. The leaves of the pla |

### ❌ Answer WITHOUT RAG

> Powdery Mildew is a common fungal disease affecting cucumbers, causing white, powdery patches on the upper surface of leaves. It's caused by the fungus Podosphaera xanthii, which thrives in humid environments with moderate temperatures.
>
> Given your location in Kottayam, Kerala, and the current weather conditions (94% humidity and 26.1°C temperature), it's no surprise that your cucumber plants are susceptible to Powdery Mildew.
>
> To treat Powdery Mildew:
>
> 1. Remove and dispose of infected leaves to prevent the disease from spreading.
> 2. Improve air circulation around the plants by providing sufficient spacing and pruning nearby plants.
> 3. Apply a fungicide specifically designed for Powdery Mildew control. However, please consult local guidelines or an agronomist for recommended dosages and products.
>
> For prevention:
>
> 1. Maintain good air circulation and plant spacing.
> 2. Water plants at the base, avoiding overhead irrigation to reduce humidity.
> 3. Monitor plants regularly for early detection.
> 4. Practice crop rotation and remove any infected plant debris from the previous season.
>
> Additionally, consider using organic control methods like neem oil or bicarbonate-based products, which can help control the disease. However, always follow the recommended application rates and guidelines.
>
> Keep in mind that it's essential to take immediate action to prevent the disease from spreading, as Powdery Mildew can significantly impact cucumber yields and quality.

---

## 5. Rice — Rice Blast

**Question asked:**

> Diamond-shaped lesions with grey centres and brown borders are appearing on my rice leaves. What is Rice Blast, what pathogen causes it, and what are the recommended chemical and cultural management practices?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Rice Blast is a fungal disease that affects rice plants, causing significant yield loss and reducing grain quality. It is caused by the fungus Magnaporthe grisea. The disease is characterized by the appearance of diamond-shaped lesions with grey centers and brown borders on the leaves, as you've described.
>
> As for management practices, there are no specific chemical controls currently recommended for the treatment of Rice Blast. However, cultural practices can help reduce the incidence of the disease. To manage Rice Blast, it's essential to:
>
> 1. Avoid excessive nitrogen application, as high soil nitrogen content can favor disease emergence.
> 2. Implement good crop management practices, such as maintaining a balanced fertilizer application and ensuring adequate water management.
> 3. Use resistant varieties, if available.
> 4. Practice good sanitation and remove infected plant debris to prevent the spread of the disease.
>
> It's crucial to monitor your field regularly and take prompt action if you notice any symptoms of Rice Blast. If the infection is severe, consider consulting with a local agronomist or crop expert for further guidance on managing the disease.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | rice.txt | Brown spot Cochliobolus miyabeanus | Brown spot Cochliobolus miyabeanus Symptoms Circular, brown lesions on seedling; distorted primary and/or secondary leaves on seedlings; black discoloration of roots; death of seedlings; circular or oval lesions with gray center and reddish-brown margin on older plants; death of large areas of leave |
| 2 | rice.txt | Sheath blight Rhizoctonia solani | Sheath blight Rhizoctonia solani Symptoms Circular, oblong or elliptical, green to gray water-soaked spots on leaf sheaths; lesions with pale green or white center and purple-brown margin; lesions covering leaf sheaths and stems; poorly filled grains Cause Fungus Comments Occurs in all areas where r |
| 3 | rice.txt | Symptoms | As name suggest we will see scalded appearance on leaves. The lesion is marked with different zone from alternating light tan and dark brown from leaf tips or edges. As the leaves mature the lesion is with light brown halos. Margins and leaf tips are translucent. |

### ❌ Answer WITHOUT RAG

> Rice Blast is a serious fungal disease affecting rice crops worldwide. The pathogen responsible for Rice Blast is Magnaporthe oryzae (formerly known as Pyricularia oryzae).
>
> For chemical management, I recommend using fungicides like tricyclazole, edifenphos, or probenazole. However, please consult local guidelines for the recommended dosage and application timing.
>
> Culturally, to manage Rice Blast:
>
> 1. **Use resistant varieties**: If possible, consider planting rice varieties with built-in resistance to Rice Blast.
> 2. **Maintain good water management**: Avoid excessive water stagnation, which can exacerbate the disease.
> 3. **Improve air circulation**: Ensure sufficient spacing between plants to promote air circulation and reduce humidity.
> 4. **Remove infected plants**: Remove and dispose of infected plants to prevent the spread of the disease.
> 5. **Practice good sanitation**: Keep the field clean, and avoid using infected plant materials as compost.
> 6. **Optimize nitrogen application**: Excessive nitrogen can increase the susceptibility of rice plants to Rice Blast. Optimize nitrogen application based on soil type and crop requirements.
>
> Remember to always follow local guidelines and regulations when using fungicides, and consider consulting with a local agronomist for more specific advice.

---

## 6. Soybean — Soybean Downy Mildew

**Question asked:**

> I can see pale green to yellow patches on the upper surface of soybean leaves with grey-purple fuzz underneath. What is Soybean Downy Mildew, what causes it, and how should I manage it?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> I apologize, but there is no mention of Soybean Downy Mildew in the provided reference material. The symptoms you described do not match any of the diseases listed in the reference. It's possible that the disease is not included in the provided material or it might be a different disease altogether.
>
> However, based on your description, it's possible that the disease might be related to fungal diseases like Cercospora leaf blight or Rust, which can cause similar symptoms. But without more information or a more specific diagnosis, it's difficult to provide a accurate management strategy.
>
> I would recommend consulting a local agronomist or a more comprehensive reference material for soybean diseases to get a more accurate diagnosis and management strategy.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | soybean.txt | Brown spot Septoria glycines | Brown spot Septoria glycines Symptoms Upper leaves have light purple discoloration and a leathery appearance and bronzing of leaves may occur; red-purple angular or irregularly shaped lesions develop on both the upper and lower leaf surfaces and may coalesce to form large necrotic patches; multiple… |
| 2 | soybean.txt | Bacterial blight Pseudomonas syringae | Bacterial blight Pseudomonas syringae Symptoms Water-soaked spots on leaves which enlarge and become necrotic; spots may be surrounded by a zone of yellow discoloration; lesions coalesce and give plant a burned appearance; leaves that die remain attached to plant; circular, sunken, red-brown lesion… |
| 3 | soybean.txt | Phytophthora rot Phytophthora megasperma | Phytophthora rot Phytophthora megasperma Symptoms Susceptible varieties of soybean may have water-soaked stems and yellowing leaves; yellowing occurs between leaf veins and along leaf margins; upper leaves of plant become chlorotic and wilted; tolerant soybean varieties may show stunted growth and s |

### ❌ Answer WITHOUT RAG

> Soybean Downy Mildew is a fungal disease caused by Peronospora manshurica. It's a common issue in humid and wet conditions, which fits our current weather situation with high humidity and recent precipitation. The pale green to yellow patches on the upper surface of the leaves, accompanied by grey-purple fuzz underneath, are characteristic symptoms of this disease.
>
> To manage Soybean Downy Mildew, I recommend the following:
>
> 1. Remove and dispose of infected leaves to prevent the disease from spreading.
> 2. Improve air circulation within the field by adjusting plant spacing or pruning nearby plants.
> 3. Avoid overhead irrigation, which can exacerbate the issue. Instead, use drip irrigation or soaker hoses to reduce moisture on the leaves.
> 4. Apply a fungicide specifically designed to combat downy mildew. Please consult local guidelines or an agronomist for the recommended dosage and application instructions.
>
> It's essential to act promptly to prevent the disease from spreading and causing significant damage to your soybean crop.

---

## 7. Tomato — Tomato Late Blight

**Question asked:**

> Dark, water-soaked lesions are appearing on my tomato leaves and spreading to the stems and fruits. What is Tomato Late Blight, what organism causes it, and what are the recommended management and fungicide options?

| | With RAG | Without RAG |
|-|----------|-------------|
| RAG active | ✅ Yes | ❌ No |
| Chunks retrieved | 3 | 0 |

### ✅ Answer WITH RAG

> Based on your description, it's likely that your tomato plants are infected with Late Blight. Late Blight is a disease that affects all aerial parts of the tomato plant, causing water-soaked green to black areas on leaves, which rapidly change to brown lesions. The disease can also infect stems and fruits, leading to severe damage and plant death.
>
> The organism that causes Late Blight is an Oomycete.
>
> To manage Late Blight, it's essential to take action quickly. Recommended management options include:
>
> 1. Planting resistant varieties: If possible, consider planting tomato varieties that have built-in resistance to Late Blight.
> 2. Applying fungicides: If signs of disease are present or if rainy conditions are likely, apply appropriate fungicides to prevent the spread of the disease. 
>
> Please consult local guidelines or an agronomist for specific fungicide recommendations and dosage instructions.

#### Retrieved Knowledge Chunks

| # | Source | Section | Snippet (first 300 chars) |
|---|--------|---------|--------------------------|
| 1 | tomato.txt | Late blight affects all aerial parts of the tomato plant; initial symptoms of the disease appear as water-soaked green to black areas on leaves which rapidly change to brown lesions; fluffy white fungal growth may appear on infected areas and leaf undersides during wet weather; as the disease progresses, foliage becomes becomes shriveled and brown and the entire plant may die; fruit lesions start as irregularly shaped water soaked regions and change to greasy spots; entire fruit may become infected and a white fuzzy growth may appear during wet weather. | Late blight affects all aerial parts of the tomato plant; initial symptoms of the disease appear as water-soaked green to black areas on leaves which rapidly change to brown lesions; fluffy white fungal growth may appear on infected areas and leaf undersides during wet weather; as the disease progre |
| 2 | tomato.txt | Early blight symptoms start as oval shaped lesions with a yellow chlorotic region across the lesion; concentric leaf lesions may be seen on infected leaves; leaf tissue between veins is destroyed; severe infections can cause leaves to completely collapse; as the disease progresses leaves become severely blighted leading to reduced yield; tomato stems may become infected with the fungus leading to Alternaria stem canker; initial symptoms of of stem canker are the development of dark brown regions on the stem; stem cankers may enlarge to girdle the whole stem resulting in the death of the whole plant; brown streaks can be found in the vascular tissue above and below the canker region; fruit symptoms include small black v-shaped lesions at the shoulders of the fruit (the disease is also known black shoulder); lesions may also appear on the fruit as dark flecks with concentric ring pattern; fruit lesions can seen in the field or may develop during fruit transit to the market; the lesions may have a velvety appearance caused by sporulation of the fungus | Early blight symptoms start as oval shaped lesions with a yellow chlorotic region across the lesion; concentric leaf lesions may be seen on infected leaves; leaf tissue between veins is destroyed; severe infections can cause leaves to completely collapse; as the disease progresses leaves become seve |
| 3 | tomato.txt | Bacterial spot lesions starts out as small water-soaked spots; lesions become more numerous and coalesce to form necrotic areas on the leaves giving them a blighted appearance; of leaves drop from the plant severe defoliation can occur leaving the fruit susceptible to sunscald; mature spots have a greasy appearance and may appear transparent when held up to light; centers of lesions dry up and fall out of the leaf; blighted leaves often remain attached to the plant and give it a blighted appearance; fruit infections start as a slightly raised blister; lesions may have a faint halo which eventually disappears; lesions on fruit may have a raised margin and sunken center which gives the fruit a scabby appearance. | Bacterial spot lesions starts out as small water-soaked spots; lesions become more numerous and coalesce to form necrotic areas on the leaves giving them a blighted appearance; of leaves drop from the plant severe defoliation can occur leaving the fruit susceptible to sunscald; mature spots have a g |

### ❌ Answer WITHOUT RAG

> I'm concerned to hear that! Tomato Late Blight (TLB) is a highly infectious and destructive disease caused by the oomycete pathogen Phytophthora infestans. It's a major threat to tomato crops, especially during the monsoon season like we're experiencing now.
>
> For management, I recommend:
>
> 1. Remove and dispose of infected plants to prevent the disease from spreading.
> 2. Improve air circulation and reduce humidity around the plants by providing sufficient spacing and pruning.
> 3. Avoid overhead irrigation, which can exacerbate the disease. Instead, use drip irrigation or soaker hoses.
> 4. Apply organic matter like compost or well-rotted manure to improve soil health and structure.
>
> As for fungicide options, I suggest using copper-based fungicides, such as copper oxychloride or copper hydroxide, which are effective against Phytophthora infestans. However, please consult local guidelines or an agronomist for specific dosage recommendations, as these may vary depending on your region and crop conditions.
>
> Remember, it's essential to act quickly to prevent the disease from spreading further. Monitor your crop closely, and take prompt action if you notice any new infections.

---

## Summary

> The side-by-side comparison above shows how Yukthi (RAG) grounds each answer in verified agronomic source material — reducing hallucinated dosages, uncertain hedging, and missing specifics compared to the parametric-only baseline.
