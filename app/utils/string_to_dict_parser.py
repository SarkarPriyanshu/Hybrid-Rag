import re

def parse_row_to_metadata(text: str) -> dict:
    """
    Parses a long string containing metadata fields in the format:
    key: value, key: value, ...
    Values may contain commas or long multi-line text.

    Returns a dictionary with clean key → value mappings.
    """

    # Define the exact fields we expect (in correct order)
    keys = [
        "setId",
        "genericName",
        "activeIngredients",
        "inactiveIngredients",
        "description",
        "indications",
        "warnings",
    ]

    # Build a regex pattern that finds "key:" and captures content
    # until the next key:
    # (setId|genericName|...): <value up to next key>
    pattern = r"({keys}):\s*(.*?)\s*(?=({keys}):|$)".format(
        keys="|".join(keys)
    )

    matches = re.findall(pattern, text, flags=re.DOTALL)

    result = {}
    for key, value, _ in matches:
        cleaned = value.strip().rstrip(",")  # remove trailing comma if any
        result[key] = cleaned

    return result


# if __name__ == "__main__":
#     strings = ["setId: 0126ea41-2cdc-4a6c-af5f-d036ef3679e5, genericName: Minoxidil, activeIngredients: MINOXIDIL 50mg, inactiveIngredients: WATER, PROPYLENE GLYCOL, HYALURONIC ACID, COCAMIDOPROPYL BETAINE, HAIR KERATIN AMINO ACIDS, AMINO ACIDS, SOURCE UNSPECIFIED, TOCOPHEROL, GLYCERIN, XANTHAN GUM, ALLANTOIN, ALOE VERA LEAF, CITRUS PARADISI SEED, REYNOUTRIA JAPONICA WHOLE, BIOTIN, ETHYLHEXYLGLYCERIN, description: Not available, indications: To regrow hair the scalp, warnings: Not available",
#         "setId: 02f7bf9c-1aa3-4734-8a3e-1f49498055d7, genericName: DEXAMETHASONE, activeIngredients: DEXAMETHASONE 0.5mg, inactiveIngredients: BENZOIC ACID, ALCOHOL, CITRIC ACID, EDETATE DISODIUM, FD&C RED NO. 40, PROPYLENE GLYCOL, WATER, SACCHARIN SODIUM, SORBITOL SOLUTION, description: Dexamethasone, USP……………………………………0.5 mg Benzoic Acid, USP……………………………………….0.1% (as a preservative)Alcohol………………………………………………………5% Artificial raspberry flavor, citric acid, edetate disodium, FD&C Red #40, propylene glycol, purified water, saccharin sodium, sorbitol solution. Glucocorticoids are adrenocortical steroids, both naturally occurring and synthetic, which are readily absorbed from the gastrointestinal tract. Dexamethasone, a synthetic adrenocortical steroid, is a white to practically white, odorless, crystalline powder. It is stable in air. It is practically insoluble in water. The molecular weight is 392.47. It is designated chemically as 9-fluoro-11β,17,21-trihydroxy-16α-methylpregna-1,4-diene-3,20-dione. The molecular formula is CHFO and the structural formula is:, indications: Not available, warnings: Not available",
#         """setId: 04577c39-a9c0-4c66-a775-4c79e2c3dea5, genericName: pioglitazone hydrochloride, activeIngredients: PIOGLITAZONE HYDROCHLORIDE 45mg, inactiveIngredients: CARBOXYMETHYLCELLULOSE CALCIUM, HYDROXYPROPYL CELLULOSE (1600000 WAMW), LACTOSE MONOHYDRATE, MAGNESIUM STEARATE, description: Pioglitazone tablets are a thiazolidinedione and an agonist for peroxisome proliferator-activated receptor (PPAR) gamma that contains an oral antidiabetic medication: pioglitazone.

        

#                             Pioglitazone [(±)-5-[[4-[2-(5-ethyl-2-pyridinyl) ethoxy] phenyl] methyl]-2,4-] thiazolidinedione monohydrochloride contains one asymmetric carbon, and the compound is synthesized and used as the racemic mixture. The two enantiomers of pioglitazone interconvert

        

#         . No differences were found in the pharmacologic activity between the two enantiomers. The structural formula is as shown: Pioglitazone hydrochloride USP is an off-white to pale yellow color powder that has a molecular formula of C

        

#         H

        

#         N

        

#         O

        

#         S•HCl and a molecular weight of 392.90 daltons. It is soluble in

        

#         - dimethylformamide, slightly soluble in anhydrous ethanol, very slightly soluble in acetone and acetonitrile, practically insoluble in water, and insoluble in ether. 

        

#                             Pioglitazone is available as a tablet for oral administration containing 15 mg, 30 mg, or 45 mg of pioglitazone (as the base) formulated with the following excipients: carboxymethylcellulose calcium, hydroxypropyl cellulose, lactose monohydrate, and magnesium stearate., indications: Not available, warnings: Not available

#         """,
#         "setId: 16b7e44d-afa0-4177-80ef-c3ae9a78a4e1, genericName: Not available, activeIngredients: Not available, inactiveIngredients: Not available, description: Sumatriptan injection USP contains sumatriptan succinate, a selective 5-HT receptor agonist. Sumatriptan succinate is chemically designated as 3-[2-(dimethyl amino) ethyl]-N-methyl-indole-5-methanesulfonamide succinate (1:1), and it has the following structure: The molecular formula is CHNOS•CHO, representing a molecular weight of 413.5. Sumatriptan succinate is a white or almost white powder that is readily soluble in water and in saline.   Sumatriptan injection USP is a clear, colorless to pale yellow, sterile, nonpyrogenic solution for subcutaneous injection.  Each 0.5 mL of sumatriptan injection, 8 mg/mL contains 5.6 mg of sumatriptan succinate equivalent to 4 mg of sumatriptan and 3.8 mg of sodium chloride, USP in Water for Injection, USP. Each 0.5 mL of sumatriptan injection, 12 mg/mL contains  8.4 mg of sumatriptan succinate equivalent to 6 mg of sumatriptan and 3.5 mg of sodium chloride, USP in Water for Injection, USP. The pH range of both solutions is approximately 4.2 to 5.3. The osmolality of both injections is 291 mOsmol., indications: Not available, warnings: Not available",
#         "setId: 1a21b07a-263c-41a4-ba0c-ad2dc5a295d7, genericName: Sennosides, Docusate Sodium, activeIngredients: DOCUSATE SODIUM 50mg, SENNOSIDES 8.6mg, inactiveIngredients: STARCH, CORN, CROSCARMELLOSE SODIUM, DIBASIC CALCIUM PHOSPHATE DIHYDRATE, FD&C BLUE NO. 1 ALUMINUM LAKE, FD&C RED NO. 40 ALUMINUM LAKE, FD&C YELLOW NO. 6 ALUMINUM LAKE, HYPROMELLOSE, UNSPECIFIED, MAGNESIUM STEARATE, MALTODEXTRIN, MICROCRYSTALLINE CELLULOSE, LIGHT MINERAL OIL, POLYETHYLENE GLYCOL 400, SODIUM BENZOATE, SODIUM LAURYL SULFATE, STEARIC ACID, TALC, TITANIUM DIOXIDE, description: Not available, indications: Not available, warnings: Not available"
#         ]
    
#     parsed_dicts = list()

#     for str in strings:
#         parsed_dicts.append(parse_row_to_metadata(str))


#     for ele in parsed_dicts:
#         print(ele)    