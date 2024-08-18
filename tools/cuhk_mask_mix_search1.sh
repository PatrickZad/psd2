for maskr in 0.1 0.16 ;  
do
    for mixr in 0.2 0.25 0.3 ;  
    do
        python tools/train_ps_net_plain.py --config-file configs/person_search/oim/def/oim_clip_simple_detboxaug_coattmaskhigh_coattoptxidmix_oimshare_sdm_miml2dfullypredVe_cuhk_30_slr_open_b10_mim05_plr_tdrop_cws.yaml  --num-gpus 1 --resume --dist-url tcp://127.0.0.1:60888 PERSON_SEARCH.REID.MASK_RATIO ${maskr} PERSON_SEARCH.REID.MIX_RATIO ${mixr} OUTPUT_DIR outputs/oim_clip/def/oim_simple_detboxaug_coattmaskhigh_coattoptxidmix_oimshare_sdm_miml2dfullypredve_cuhk_30_slr_open_b10_05mr_plr_tdrop_cws_${maskr}mask_${mixr}mix
    done
done