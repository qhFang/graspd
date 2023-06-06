import os
import subprocess
import argparse
import numpy as np
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)    
parser.add_argument('--start', type=int, default=0,
                    help='absolute path of pretrained model')
cmd_args = parser.parse_args()
start = cmd_args.start
start= 0
'''
for root, dirs, files in os.walk('/apdcephfs/private_qihangfang/Codes/IBS-Grasping/Grasp_Dataset_v4/'):
    dirs = np.unique(np.array(dirs))
    break

test_split = []
for dir in dirs:
    dataset = dir.split('_')[0]
    if dataset =='ycb' or dataset=='bigbird':
        test_split.append(dir)

print(test_split)

subprocessess = []
'''

test_split = ['bigbird_3m_high_tack_spray_adhesive_scaled', 'bigbird_advil_liqui_gels_scaled', 'bigbird_band_aid_clear_strips_scaled', 'bigbird_band_aid_sheer_strips_scaled', 'bigbird_blue_clover_baby_toy_scaled', 'bigbird_bumblebee_albacore_scaled', 'bigbird_campbells_chicken_noodle_soup_scaled', 'bigbird_campbells_soup_at_hand_creamy_tomato_scaled', 'bigbird_canon_ack_e10_box_scaled', 'bigbird_chewy_dipps_chocolate_chip_scaled', 'bigbird_chewy_dipps_peanut_butter_scaled', 'bigbird_cholula_chipotle_hot_sauce_scaled', 'bigbird_clif_crunch_chocolate_chip_scaled', 'bigbird_clif_crunch_peanut_butter_scaled', 'bigbird_clif_crunch_white_chocolate_macademia_nut_scaled', 'bigbird_clif_z_bar_chocolate_chip_scaled', 'bigbird_clif_zbar_chocolate_brownie_scaled', 'bigbird_coffee_mate_french_vanilla_scaled', 'bigbird_colgate_cool_mint_scaled', 'bigbird_crayola_24_crayons_scaled', 'bigbird_crayola_yellow_green_scaled', 'bigbird_crest_complete_minty_fresh_scaled', 'bigbird_crystal_hot_sauce_scaled', 'bigbird_cup_noodles_chicken_scaled', 'bigbird_cup_noodles_shrimp_picante_scaled', 'bigbird_dove_beauty_cream_bar_scaled', 'bigbird_dove_go_fresh_burst_scaled', 'bigbird_dove_pink_scaled', 'bigbird_expo_marker_red_scaled', 'bigbird_fruit_by_the_foot_scaled', 'bigbird_haagen_dazs_butter_pecan_scaled', 'bigbird_haagen_dazs_cookie_dough_scaled', 'bigbird_hersheys_cocoa_scaled', 'bigbird_hunts_paste_scaled', 'bigbird_hunts_sauce_scaled', 'bigbird_ikea_table_leg_blue_scaled', 'bigbird_krylon_crystal_clear_scaled', 'bigbird_krylon_matte_finish_scaled', 'bigbird_krylon_short_cuts_scaled', 'bigbird_mom_to_mom_butternut_squash_pear_scaled', 'bigbird_mom_to_mom_sweet_potato_corn_apple_scaled', 'bigbird_nature_valley_crunchy_oats_n_honey_scaled', 'bigbird_nature_valley_crunchy_variety_pack_scaled', 'bigbird_nature_valley_gluten_free_roasted_nut_crunch_almond_crunch_scaled', 'bigbird_nature_valley_granola_thins_dark_chocolate_scaled', 'bigbird_nature_valley_soft_baked_oatmeal_squares_cinnamon_brown_sugar_scaled', 'bigbird_nature_valley_soft_baked_oatmeal_squares_peanut_butter_scaled', 'bigbird_nature_valley_sweet_and_salty_nut_almond_scaled', 'bigbird_nature_valley_sweet_and_salty_nut_cashew_scaled', 'bigbird_nature_valley_sweet_and_salty_nut_peanut_scaled', 'bigbird_nature_valley_sweet_and_salty_nut_roasted_mix_nut_scaled', 'bigbird_nice_honey_roasted_almonds_scaled', 'bigbird_nutrigrain_apple_cinnamon_scaled', 'bigbird_nutrigrain_blueberry_scaled', 'bigbird_nutrigrain_cherry_scaled', 'bigbird_nutrigrain_chotolatey_crunch_scaled', 'bigbird_nutrigrain_fruit_crunch_apple_cobbler_scaled', 'bigbird_nutrigrain_fruit_crunch_strawberry_parfait_scaled', 'bigbird_nutrigrain_harvest_blueberry_bliss_scaled', 'bigbird_nutrigrain_harvest_country_strawberry_scaled', 'bigbird_nutrigrain_toffee_crunch_chocolatey_toffee_scaled', 'bigbird_pepto_bismol_scaled', 'bigbird_pop_secret_butter_scaled', 'bigbird_pop_secret_light_butter_scaled', 'bigbird_pop_tarts_strawberry_scaled', 'bigbird_pringles_bbq_scaled', 'bigbird_progresso_new_england_clam_chowder_scaled', 'bigbird_quaker_big_chewy_chocolate_chip_scaled', 'bigbird_quaker_big_chewy_peanut_butter_chocolate_chip_scaled', 'bigbird_quaker_chewy_dipps_peanut_butter_chocolate_scaled', 'bigbird_quaker_chewy_low_fat_chocolate_chunk_scaled', 'bigbird_quaker_chewy_peanut_butter_chocolate_chip_scaled', 'bigbird_quaker_chewy_peanut_butter_scaled', 'bigbird_quaker_chewy_smores_scaled', 'bigbird_red_bull_scaled', 'bigbird_red_cup_scaled', 'bigbird_softsoap_gold_scaled', 'bigbird_softsoap_white_scaled', 'bigbird_south_beach_good_to_go_peanut_butter_scaled', 'bigbird_spam_scaled', 'bigbird_suave_sweet_guava_nectar_body_wash_scaled', 'bigbird_tapatio_hot_sauce_scaled', 'bigbird_v8_fusion_peach_mango_scaled', 'bigbird_v8_fusion_strawberry_banana_scaled', 'bigbird_vo5_extra_body_volumizing_shampoo_scaled', 'bigbird_vo5_split_ends_anti_breakage_shampoo_scaled', 'bigbird_vo5_tea_therapy_healthful_green_tea_smoothing_shampoo_scaled', 'bigbird_white_rain_sensations_apple_blossom_hydrating_body_wash_scaled', 'bigbird_white_rain_sensations_ocean_mist_hydrating_body_wash_scaled', 'bigbird_white_rain_sensations_ocean_mist_hydrating_conditioner_scaled', 'bigbird_zilla_night_black_heat_scaled', 'ycb_002_master_chef_can_scaled', 'ycb_003_cracker_box_scaled', 'ycb_004_sugar_box_scaled', 'ycb_008_pudding_box_scaled', 'ycb_010_potted_meat_can_scaled', 'ycb_011_banana_scaled', 'ycb_012_strawberry_scaled', 'ycb_013_apple_scaled', 'ycb_014_lemon_scaled', 'ycb_015_peach_scaled', 'ycb_017_orange_scaled', 'ycb_018_plum_scaled', 'ycb_021_bleach_cleanser_scaled', 'ycb_033_spatula_scaled', 'ycb_036_wood_block_scaled', 'ycb_037_scissors_scaled', 'ycb_038_padlock_scaled', 'ycb_040_large_marker_scaled', 'ycb_043_phillips_screwdriver_scaled', 'ycb_044_flat_screwdriver_scaled', 'ycb_048_hammer_scaled', 'ycb_050_medium_clamp_scaled', 'ycb_052_extra_large_clamp_scaled', 'ycb_054_softball_scaled', 'ycb_055_baseball_scaled', 'ycb_056_tennis_ball_scaled', 'ycb_057_racquetball_scaled', 'ycb_058_golf_ball_scaled', 'ycb_061_foam_brick_scaled', 'ycb_063-a_marbles_scaled', 'ycb_063-b_marbles_scaled', 'ycb_065-a_cups_scaled', 'ycb_065-b_cups_scaled', 'ycb_065-c_cups_scaled', 'ycb_065-e_cups_scaled', 'ycb_065-f_cups_scaled', 'ycb_065-h_cups_scaled', 'ycb_070-a_colored_wood_blocks_scaled', 'ycb_070-b_colored_wood_blocks_scaled', 'ycb_072-b_toy_airplane_scaled', 'ycb_072-c_toy_airplane_scaled', 'ycb_072-d_toy_airplane_scaled', 'ycb_072-e_toy_airplane_scaled', 'ycb_073-a_lego_duplo_scaled', 'ycb_073-b_lego_duplo_scaled', 'ycb_073-c_lego_duplo_scaled', 'ycb_073-d_lego_duplo_scaled', 'ycb_073-f_lego_duplo_scaled', 'ycb_073-g_lego_duplo_scaled', 'ycb_077_rubiks_cube_scaled']
print(len(test_split))

ibs_train_split = ['gd_avocado_poisson_000_scaled', 'gd_avocado_poisson_001_scaled', 'gd_banana_poisson_000_scaled', 'gd_banana_poisson_001_scaled', 'gd_banana_poisson_002_scaled', 'gd_banana_poisson_003_scaled', 'gd_banana_poisson_004_scaled', 'gd_banana_poisson_005_scaled', 'gd_binder_poisson_003_scaled', 'gd_bottle_new_poisson_000_scaled', 'gd_bowling_pin_poisson_000_scaled', 'gd_box_poisson_000_scaled', 'gd_box_poisson_001_scaled', 'gd_box_poisson_002_scaled', 'gd_box_poisson_003_scaled', 'gd_box_poisson_005_scaled', 'gd_box_poisson_006_scaled', 'gd_box_poisson_008_scaled', 'gd_box_poisson_009_scaled', 'gd_box_poisson_010_scaled', 'gd_box_poisson_011_scaled', 'gd_box_poisson_013_scaled', 'gd_box_poisson_014_scaled', 'gd_box_poisson_015_scaled', 'gd_box_poisson_016_scaled', 'gd_box_poisson_017_scaled', 'gd_box_poisson_018_scaled', 'gd_box_poisson_019_scaled', 'gd_box_poisson_020_scaled', 'gd_box_poisson_021_scaled', 'gd_box_poisson_022_scaled', 'gd_box_poisson_023_scaled', 'gd_box_poisson_024_scaled', 'gd_box_poisson_025_scaled', 'gd_box_poisson_026_scaled', 'gd_camera_poisson_000_scaled', 'gd_camera_poisson_002_scaled', 'gd_camera_poisson_003_scaled', 'gd_camera_poisson_004_scaled', 'gd_camera_poisson_006_scaled', 'gd_camera_poisson_007_scaled', 'gd_camera_poisson_010_scaled', 'gd_camera_poisson_011_scaled', 'gd_camera_poisson_012_scaled', 'gd_camera_poisson_013_scaled', 'gd_camera_poisson_014_scaled', 'gd_camera_poisson_015_scaled', 'gd_camera_poisson_016_scaled', 'gd_camera_poisson_017_scaled', 'gd_camera_poisson_019_scaled', 'gd_can_poisson_000_scaled', 'gd_can_poisson_002_scaled', 'gd_can_poisson_003_scaled', 'gd_can_poisson_006_scaled', 'gd_can_poisson_007_scaled', 'gd_can_poisson_009_scaled', 'gd_can_poisson_010_scaled', 'gd_can_poisson_011_scaled', 'gd_can_poisson_012_scaled', 'gd_can_poisson_014_scaled', 'gd_can_poisson_015_scaled', 'gd_can_poisson_016_scaled', 'gd_can_poisson_018_scaled', 'gd_can_poisson_019_scaled', 'gd_can_poisson_020_scaled', 'gd_can_poisson_022_scaled', 'gd_champagne_glass_poisson_001_scaled', 'gd_croissant_poisson_000_scaled', 'gd_cucumber_poisson_000_scaled', 'gd_detergent_bottle_poisson_001_scaled', 'gd_detergent_bottle_poisson_002_scaled', 'gd_detergent_bottle_poisson_003_scaled', 'gd_detergent_bottle_poisson_004_scaled', 'gd_detergent_bottle_poisson_006_scaled', 'gd_detergent_bottle_poisson_007_scaled', 'gd_donut_poisson_000_scaled', 'gd_donut_poisson_001_scaled', 'gd_donut_poisson_002_scaled', 'gd_donut_poisson_003_scaled', 'gd_donut_poisson_004_scaled', 'gd_donut_poisson_005_scaled', 'gd_donut_poisson_006_scaled', 'gd_donut_poisson_007_scaled', 'gd_donut_poisson_008_scaled', 'gd_donut_poisson_009_scaled', 'gd_drill_poisson_000_scaled', 'gd_dumpbell_poisson_000_scaled', 'gd_dumpbell_poisson_001_scaled', 'gd_egg_poisson_000_scaled', 'gd_egg_poisson_001_scaled', 'gd_egg_poisson_002_scaled', 'gd_egg_poisson_003_scaled', 'gd_egg_poisson_004_scaled', 'gd_egg_poisson_005_scaled', 'gd_egg_poisson_006_scaled', 'gd_egg_poisson_007_scaled', 'gd_egg_poisson_008_scaled', 'gd_egg_poisson_009_scaled', 'gd_egg_poisson_010_scaled', 'gd_egg_poisson_011_scaled', 'gd_egg_poisson_012_scaled', 'gd_figurine_poisson_005_scaled', 'gd_flashlight_poisson_000_scaled', 'gd_flashlight_poisson_002_scaled', 'gd_flashlight_poisson_003_scaled', 'gd_flashlight_poisson_005_scaled', 'gd_flashlight_poisson_007_scaled', 'gd_flashlight_poisson_008_scaled', 'gd_flashlight_poisson_010_scaled', 'gd_flashlight_poisson_013_scaled', 'gd_flashlight_poisson_014_scaled', 'gd_hammer_poisson_002_scaled', 'gd_hammer_poisson_004_scaled', 'gd_hammer_poisson_005_scaled', 'gd_hammer_poisson_006_scaled', 'gd_hammer_poisson_008_scaled', 'gd_hammer_poisson_009_scaled', 'gd_hammer_poisson_011_scaled', 'gd_hammer_poisson_012_scaled', 'gd_hammer_poisson_013_scaled', 'gd_hammer_poisson_015_scaled', 'gd_hammer_poisson_016_scaled', 'gd_hammer_poisson_017_scaled', 'gd_hammer_poisson_018_scaled', 'gd_hammer_poisson_020_scaled', 'gd_hammer_poisson_024_scaled', 'gd_hammer_poisson_027_scaled', 'gd_hammer_poisson_029_scaled', 'gd_hammer_poisson_031_scaled', 'gd_hammer_poisson_034_scaled', 'gd_hammer_poisson_035_scaled', 'gd_jar_poisson_001_scaled', 'gd_jar_poisson_002_scaled', 'gd_jar_poisson_003_scaled', 'gd_jar_poisson_005_scaled', 'gd_jar_poisson_006_scaled', 'gd_jar_poisson_007_scaled', 'gd_jar_poisson_008_scaled', 'gd_jar_poisson_013_scaled', 'gd_jar_poisson_014_scaled', 'gd_jar_poisson_017_scaled', 'gd_jar_poisson_018_scaled', 'gd_jar_poisson_019_scaled', 'gd_jar_poisson_022_scaled', 'gd_jar_poisson_024_scaled', 'gd_lemon_poisson_000_scaled', 'gd_lemon_poisson_001_scaled', 'gd_lemon_poisson_002_scaled', 'gd_lemon_poisson_003_scaled', 'gd_lemon_poisson_004_scaled', 'gd_light_bulb_poisson_000_scaled', 'gd_light_bulb_poisson_001_scaled', 'gd_light_bulb_poisson_003_scaled', 'gd_light_bulb_poisson_004_scaled', 'gd_light_bulb_poisson_005_scaled', 'gd_light_bulb_poisson_006_scaled', 'gd_light_bulb_poisson_007_scaled', 'gd_light_bulb_poisson_008_scaled', 'gd_light_bulb_poisson_009_scaled', 'gd_light_bulb_poisson_010_scaled', 'gd_light_bulb_poisson_011_scaled', 'gd_lime_poisson_000_scaled', 'gd_lime_poisson_001_scaled', 'gd_mug_new_poisson_001_scaled', 'gd_mushroom_poisson_000_scaled', 'gd_mushroom_poisson_002_scaled', 'gd_mushroom_poisson_003_scaled', 'gd_mushroom_poisson_004_scaled', 'gd_mushroom_poisson_005_scaled', 'gd_mushroom_poisson_006_scaled', 'gd_mushroom_poisson_007_scaled', 'gd_mushroom_poisson_008_scaled', 'gd_mushroom_poisson_010_scaled', 'gd_mushroom_poisson_011_scaled', 'gd_mushroom_poisson_012_scaled', 'gd_mushroom_poisson_013_scaled', 'gd_mushroom_poisson_014_scaled', 'gd_orange_poisson_000_scaled', 'gd_orange_poisson_001_scaled', 'gd_pan_poisson_017_scaled', 'gd_pitcher_poisson_000_scaled', 'gd_pitcher_poisson_001_scaled', 'gd_pitcher_poisson_005_scaled', 'gd_pliers_poisson_000_scaled', 'gd_pliers_poisson_002_scaled', 'gd_pliers_poisson_003_scaled', 'gd_pliers_poisson_004_scaled', 'gd_pliers_poisson_005_scaled', 'gd_pliers_poisson_010_scaled', 'gd_pliers_poisson_012_scaled', 'gd_pliers_poisson_014_scaled', 'gd_pliers_poisson_015_scaled', 'gd_pliers_poisson_017_scaled', 'gd_pumpkin_poisson_000_scaled', 'gd_rubber_duck_poisson_000_scaled', 'gd_rubber_duck_poisson_001_scaled', 'gd_rubber_duck_poisson_002_scaled', 'gd_rubik_cube_poisson_004_scaled', 'gd_rubik_cube_poisson_005_scaled', 'gd_saucepan_poisson_000_scaled', 'gd_screwdriver_poisson_000_scaled', 'gd_screwdriver_poisson_002_scaled', 'gd_screwdriver_poisson_003_scaled', 'gd_screwdriver_poisson_006_scaled', 'gd_screwdriver_poisson_009_scaled', 'gd_screwdriver_poisson_011_scaled', 'gd_screwdriver_poisson_013_scaled', 'gd_screwdriver_poisson_016_scaled', 'gd_screwdriver_poisson_019_scaled', 'gd_screwdriver_poisson_020_scaled', 'gd_screwdriver_poisson_021_scaled', 'gd_screwdriver_poisson_022_scaled', 'gd_screwdriver_poisson_023_scaled', 'gd_screwdriver_poisson_024_scaled', 'gd_shampoo_new_poisson_000_scaled', 'gd_shampoo_new_poisson_001_scaled', 'gd_shampoo_new_poisson_002_scaled', 'gd_shampoo_poisson_001_scaled', 'gd_spatula_poisson_001_scaled', 'gd_spatula_poisson_002_scaled', 'gd_spray_bottle_poisson_000_scaled', 'gd_spray_bottle_poisson_002_scaled', 'gd_spray_bottle_poisson_003_scaled', 'gd_spray_can_poisson_000_scaled', 'gd_spray_can_poisson_003_scaled', 'gd_spray_can_poisson_005_scaled', 'gd_squash_poisson_000_scaled', 'gd_stapler_poisson_000_scaled', 'gd_stapler_poisson_002_scaled', 'gd_stapler_poisson_003_scaled', 'gd_stapler_poisson_004_scaled', 'gd_stapler_poisson_007_scaled', 'gd_stapler_poisson_008_scaled', 'gd_stapler_poisson_010_scaled', 'gd_stapler_poisson_011_scaled', 'gd_stapler_poisson_012_scaled', 'gd_stapler_poisson_017_scaled', 'gd_stapler_poisson_018_scaled', 'gd_stapler_poisson_022_scaled', 'gd_stapler_poisson_023_scaled', 'gd_stapler_poisson_024_scaled', 'gd_stapler_poisson_025_scaled', 'gd_stapler_poisson_028_scaled', 'gd_starfruit_poisson_000_scaled', 'gd_starfruit_poisson_002_scaled', 'gd_sweet_corn_poisson_000_scaled', 'gd_tape_poisson_000_scaled', 'gd_tape_poisson_002_scaled', 'gd_tape_poisson_003_scaled', 'gd_tape_poisson_004_scaled', 'gd_tape_poisson_005_scaled', 'gd_tennis_ball_poisson_000_scaled', 'gd_tetra_pak_poisson_000_scaled', 'gd_tetra_pak_poisson_001_scaled', 'gd_tetra_pak_poisson_002_scaled', 'gd_tetra_pak_poisson_003_scaled', 'gd_tetra_pak_poisson_004_scaled', 'gd_tetra_pak_poisson_006_scaled', 'gd_tetra_pak_poisson_007_scaled', 'gd_tetra_pak_poisson_008_scaled', 'gd_tetra_pak_poisson_009_scaled', 'gd_tetra_pak_poisson_010_scaled', 'gd_tetra_pak_poisson_013_scaled', 'gd_tetra_pak_poisson_015_scaled', 'gd_tetra_pak_poisson_016_scaled', 'gd_tetra_pak_poisson_017_scaled', 'gd_tetra_pak_poisson_018_scaled', 'gd_tetra_pak_poisson_019_scaled', 'gd_tetra_pak_poisson_020_scaled', 'gd_tetra_pak_poisson_021_scaled', 'gd_tetra_pak_poisson_022_scaled', 'gd_tetra_pak_poisson_023_scaled', 'gd_tetra_pak_poisson_024_scaled', 'gd_tetra_pak_poisson_025_scaled', 'gd_toaster_poisson_000_scaled', 'gd_toaster_poisson_006_scaled', 'gd_toilet_paper_poisson_000_scaled', 'gd_toilet_paper_poisson_001_scaled', 'gd_toilet_paper_poisson_002_scaled', 'gd_toilet_paper_poisson_003_scaled', 'gd_toilet_paper_poisson_004_scaled', 'gd_toilet_paper_poisson_005_scaled', 'gd_toilet_paper_poisson_006_scaled', 'gd_toilet_paper_poisson_007_scaled', 'gd_toilet_paper_poisson_008_scaled', 'gd_tomato_poisson_000_scaled', 'gd_toothpaste_tube_poisson_003_scaled', 'gd_toy_poisson_001_scaled', 'gd_toy_poisson_004_scaled', 'gd_toy_poisson_005_scaled', 'gd_toy_poisson_006_scaled', 'gd_toy_poisson_008_scaled', 'gd_toy_poisson_022_scaled', 'gd_toy_poisson_024_scaled', 'gd_toy_poisson_025_scaled', 'gd_watering_can_poisson_001_scaled', 'gd_watering_can_poisson_003_scaled', 'gd_zucchini_poisson_001_scaled', 'kit_BakingSoda_scaled', 'kit_BakingVanilla_scaled', 'kit_BathDetergent_scaled', 'kit_BlueSaltCube_scaled', 'kit_BroccoliSoup_scaled', 'kit_ChickenSoup_scaled', 'kit_ChocMarshmallows_scaled', 'kit_ChocSticks2_scaled', 'kit_ChocSticks_scaled', 'kit_ChocoIcing_scaled', 'kit_ChocolateBars_scaled', 'kit_ChoppedTomatoes_scaled', 'kit_CleaningCloths_scaled', 'kit_CoffeeBox_scaled', 'kit_CoffeeCookies_scaled', 'kit_CoffeeFilters2_scaled', 'kit_CoffeeFilters_scaled', 'kit_CokePlasticLarge_scaled', 'kit_CokePlasticSmallGrasp_scaled', 'kit_CokePlasticSmall_scaled', 'kit_CoughDropsBerries_scaled', 'kit_CoughDropsHoney_scaled', 'kit_CoughDropsLemon_scaled', 'kit_Deodorant_scaled', 'kit_DropsCherry_scaled', 'kit_DropsOrange_scaled', 'kit_Fish_scaled', 'kit_FruitBars_scaled', 'kit_GreenSaltCylinder_scaled', 'kit_HamburgerSauce_scaled', 'kit_Heart1_scaled', 'kit_HerbSalt_scaled', 'kit_HeringTin_scaled', 'kit_HotPot2_scaled', 'kit_HotPot_scaled', 'kit_HygieneSpray_scaled', 'kit_InstantDumplings_scaled', 'kit_InstantSauce2_scaled', 'kit_InstantSoup_scaled', 'kit_InstantTomatoSoup_scaled', 'kit_KnaeckebrotRye_scaled', 'kit_Knaeckebrot_scaled', 'kit_KoalaCandy_scaled', 'kit_LivioClassicOil_scaled', 'kit_MashedPotatoes_scaled', 'kit_MelforBottle_scaled', 'kit_MilkRice_scaled', 'kit_Moon_scaled', 'kit_NutellaGo_scaled', 'kit_OrangeMarmelade_scaled', 'kit_OrgFruitTea_scaled', 'kit_OrgHerbTea_scaled', 'kit_PatchesSensitive_scaled', 'kit_Patches_scaled', 'kit_PotatoeDumplings_scaled', 'kit_PotatoeSticks_scaled', 'kit_PowderedSugarMill_scaled', 'kit_Rice_scaled', 'kit_RuskWholemeal_scaled', 'kit_Rusk_scaled', 'kit_SardinesCan_scaled', 'kit_SauceThickener_scaled', 'kit_SauerkrautSmall_scaled', 'kit_Seal_scaled', 'kit_Shampoo_scaled', 'kit_ShowerGel_scaled', 'kit_SoftCakeOrange_scaled', 'kit_SoftCheese_scaled', 'kit_StrawberryPorridge_scaled', 'kit_Sweetener_scaled', 'kit_TomatoHerbSauce_scaled', 'kit_VitalisCereals_scaled', 'kit_YellowSaltCube2_scaled', 'kit_YellowSaltCube_scaled', 'kit_YellowSaltCylinderSmall_scaled']
print(len(ibs_train_split))
#exit()

#finish_list = np.loadtxt('/apdcephfs/private_qihangfang/finish_gene_shadow', dtype=str).tolist()
#print(finish_list)
finish_list = []
'''
for j, dir in enumerate(ibs_train_split):
    
    if not os.path.exists(f'/apdcephfs/private_qihangfang/ibsshadow/{dir}'):
        num = 0
        #print(j // 6 * 6)
    else:
        f = np.loadtxt(f'/apdcephfs/private_qihangfang/ibsshadow/{dir}')
        if len(f.shape) == 1:
            num = 1
        else:
            num = f.shape[0]


    #if num == 15:
    #    print(j//6*6)
    if num == 15:
        finish_list.append(dir)
    #a = f[:, -8:]
    #p = f[:, 1: -8]
    #print(p.shape)
    #succ_num = 0
    #for i in range(15):
    #    if a[i, 0] < 5e-4 and a[i, 1] < 0.03 and (a[i, -5:] < 0.005).sum() >=2:
    #        succ_num += 1
    #print(dir, j*15, succ_num)
    #print((a[:, 0] < 5e-4))
    #print((a[:, 1] < 0.03))
    #print((a[:, -5:] < 0.005).sum(axis=1) >= 2)
    #print((a[:, 0] < 5e-4) * (a[:, 1] < 0.03) * ((a[:, -5:] < 0.005).sum(axis=1) >= 2))
    #idx = (a[:, 0] < 5e-4) * (a[:, 1] < 0.03) * ((a[:, -5:] < 0.005).sum(axis=1) >= 2)
    #if (idx.sum() == 0):
    #    print(dir)
    #    exit()
    #p = p[idx]
    #print(p.shape)
    #np.savetxt(f'/apdcephfs/share_1330077/qihangfang/Data/ibs/{dir}/target_pose_fc.txt', p)


    #nums.append(a)

    #print(i // 5 * 5, train_split[i], a.shape[0])

#nums = np.concatenate(nums, axis=0)

#np.savetxt('/apdcephfs/private_qihangfang/nums.txt', nums)

#print(len(finish_list))
'''
try:
    os.makedirs('/apdcephfs/share_1330077/qihangfang/network/rlisojointlocal/script')
except OSError:
    pass
'''
for i in range(0, 141, 10):
    f = open(f'/apdcephfs/share_1330077/qihangfang/network/rldagger/script/{i}.sh', 'w')
    cmd = f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/pytorch-a2c-ppo-acktr-gail/evaluation_copy.py --seed {3047+i}'

    #for j in range(10):
    #    if i + j >= 141:
    #        break
    #    cmd += f' {test_split[(3047+i+j) % len(test_split)]}'
    #    print(cmd)

    f.write(cmd)
    f.close()
exit()
'''



for i in range(0, 141, 10):
    f = open(f'/apdcephfs/share_1330077/qihangfang/network/rlisojointlocal/script/{i}.sh', 'w')
    #f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/pytorch-a2c-ppo-acktr-gail/evaluation_copy.py --env_name FormClosureIBSLazyEnv --seed {3047+i}\n')
    
    for j in range(10):
        if i + j >= 141:
            break
        if i + j == 140 or j == 9 or j == 8 or j == 4:
            f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/voxel.py collector_config={test_split[(3047+i+j) % len(test_split)]}\n')
            #f.write(f'source /apdcephfs/private_qihangfang/Codes/IBS-Grasping/devel/setup.bash ; sh /apdcephfs/private_qihangfang/Codes/IBS-Grasping/start.sh ; rosrun ibs_env main.py --model_name fine --obj_name {test_split[(3047+i+j) % len(test_split)]}\n')
            #f.write(f'python /apdcephfs/private_qihangfang/Codes/IBS-Grasping/src/ibs_env/scripts/main.py --model_name fine --obj_name {test_split[(3047+i+j) % len(test_split)]}\n')

        else:
            f.write(f'nohup /opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/voxel.py collector_config={test_split[(3047+i+j) % len(test_split)]} >/dev/null 2>&1 &\n')
            #f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/sha2.py collector_config={test_split[(3047+i+j) % len(test_split)]}\n')

    f.close()    
exit()

for i, dir in enumerate(test_split):
    #if i >= start + 6:
    #    start += 6
    if i % 6 == 0:
        start = i
    
    #    continue
    #elif i % 6 < 5:
    #    continue

    #if dir in finish_list:
    #    pass
    #else:
    #    continue
    
        
    #cmds = []
    
    #if len(cmds) == 0:
    #    continue
    #print(start)
         
    f = open(f'/apdcephfs/private_qihangfang/script2/{start+144}.sh', 'a')
    #for j in range(len(cmds)):
    #    if j % 2 == 1 or j == len(cmds) - 1:
    #        f.write(f'{cmds[j]}')
    #    else:
    #        f.write(f'nohup {cmds[j]} >/dev/null 2>&1 &\n')
    #f.close()

    #for j in range(start, i+1):
    #    if not ibs_train_split[j] in finish_list:
    #        print(ibs_train_split[j])
    #        cmds.append(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/collect_shadow.py collector_config={ibs_train_split[j]}\n')
    #        #f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/collect_shadow.py collector_config={ibs_train_split[j]}\n')

  
            
    #if i == start + 1 or i == start + 3 or i == start + 5 or i == len(ibs_train_split) - 1:
    if i > start + 2 and i == start + 2 or i == start + 5 or i == len(ibs_train_split) - 1:
        f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/graspd.py collector_config={dir}\n')
    elif i > start + 2:
        f.write(f'nohup /opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/graspd.py collector_config={dir} >/dev/null 2>&1 &\n')

    if i == start + 5 or i == len(ibs_train_split) - 1:
        f.write(f'/opt/conda/envs/py36/bin/python /apdcephfs/private_qihangfang/Codes/Form-closure/graspd/grasping/scripts/check_alive.py --start {start}')
    