def get_translation_dict(cat_name_translation):
    portuguese_cat_names = cat_name_translation.to_dict()['product_category_name']
    english_cat_names = cat_name_translation.to_dict()['product_category_name_english']
    translate_dict = {}

    for p_key in portuguese_cat_names:
        if portuguese_cat_names[p_key] not in translate_dict:
            translate_dict[portuguese_cat_names[p_key]] = english_cat_names[p_key]

    return translate_dict
