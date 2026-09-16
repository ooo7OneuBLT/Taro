# -*- coding: utf-8 -*-
"""Google Scanned Objects（CC-BY 4.0）から選んだモデルを Gazebo Fuel から取得する（F2-49・2026-09-02）。

    .venv/Scripts/python.exe F/scripts/f_gso_fetch.py
保存先: F/assets/gso/<モデル名>/（meshes/model.obj, materials/textures/texture.png など）
"""
import os, sys, io, json, zipfile, urllib.request
sys.stdout.reconfigure(encoding="utf-8")
os.chdir(r"C:\claude\AI\Taro")

# 語 → 訓練/テストに使う候補（3〜5体以上ある語だけ。名簿から手で選別）
SELECT = {
    "くつ": ["Womens_Hikerfish_Boot_in_Black_Leopard_ridcCWsv8rW", "Womens_Bluefish_2Eye_Boat_Shoe_in_Tan",
             "Womens_Bluefish_2Eye_Boat_Shoe_in_White_Tumbled_YG44xIePRHw", "Mens_Gold_Cup_ASV_Capetown_Penny_Loafer_in_Black_GkQBKqABeQN",
             "Mens_Striper_Sneaker_in_White_rnp8HUli59"],
    "コップ": ["Threshold_Porcelain_Coffee_Mug_All_Over_Bead_White", "Room_Essentials_Mug_White_Yellow", "Cole_Hardware_Mug_Classic_Blue",
               "ACE_Coffee_Mug_Kristen_16_oz_cup", "Ecoforms_Cup_B4_SAN"],
    "おさら": ["Threshold_Salad_Plate_Square_Rim_Porcelain", "Threshold_Dinner_Plate_Square_Rim_White_Porcelain",
               "Threshold_Bistro_Ceramic_Dinner_Plate_Ruby_Ring", "Room_Essentials_Salad_Plate_Turquoise", "Ecoforms_Plate_S20Avocado"],
    "ボウル": ["Threshold_Porcelain_Serving_Bowl_Coupe_White", "Threshold_Bead_Cereal_Bowl_White", "Room_Essentials_Bowl_Turquiose",
               "Now_Designs_Bowl_Akita_Black", "Cole_Hardware_Bowl_Scirocco_YellowBlue"],
    "つみき": ["Granimals_20_Wooden_ABC_Blocks_Wagon", "CASTLE_BLOCKS", "Wooden_ABC_123_Blocks_50_pack", "FAIRY_TALE_BLOCKS", "50_BLOCKS"],
    "かばん": ["Jansport_School_Backpack_Blue_Streak", "Olive_Kids_Trains_Planes_Trucks_Bogo_Backpack", "Olive_Kids_Mermaids_Pack_n_Snack_Backpack",
               "Olive_Kids_Birdie_Sidekick_Backpack", "US_Army_Stash_Lunch_Bag"],
    "きょうりゅう": ["Schleich_Spinosaurus_Action_Figure", "Schleich_Allosaurus", "Schleich_Therizinosaurus_ln9cruulPqc",
                     "Great_Dinos_Triceratops_Toy", "Dino_3", "Dino_4", "Dino_5"],
    "くるま": ["Vtech_Cruise_Learn_Car_25_Years", "FIRE_TRUCK", "CITY_TAXI_POLICE_CAR", "BABY_CAR"],
    "バス": ["Sonny_School_Bus", "SORTING_BUS", "SCHOOL_BUS"],
}


def fetch(name, dst="F/assets/gso"):
    out = os.path.join(dst, name)
    if os.path.exists(os.path.join(out, "meshes", "model.obj")):
        return "済"
    os.makedirs(out, exist_ok=True)
    url = "https://fuel.gazebosim.org/1.0/GoogleResearch/models/%s.zip" % urllib.request.quote(name)
    zpath = os.path.join(out, "model.zip")
    try:
        urllib.request.urlretrieve(url, zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(out)
        os.remove(zpath)
        return "取得"
    except Exception as e:
        return "失敗: %s" % str(e)[:60]


if __name__ == "__main__":
    log = {}
    for jp, names in SELECT.items():
        for n in names:
            r = fetch(n)
            log[n] = r
            print("%s %-60s %s" % (jp, n, r), flush=True)
    io.open("F/assets/gso/_選定.json", "w", encoding="utf-8").write(json.dumps(SELECT, ensure_ascii=False, indent=1))
    print("完了:", sum(1 for v in log.values() if v in ("取得", "済")), "/", len(log))
