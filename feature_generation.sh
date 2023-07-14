


python /home/jl4mc/gate/gate/feature/generate_icps_scores.py --fastadir dataset/CASP15_inhouse/human_dataset/fasta/ --outdir dataset/CASP15_inhouse/human_dataset/cdpred/ --modeldir dataset/CASP15_inhouse/human_dataset/models/ --pairwise_dir dataset/CASP15_inhouse/human_dataset/pairwise/
python gate/feature/generate_interface_model_size.py --fastadir dataset/CASP15_inhouse/human_dataset/fasta/ --outdir dataset/CASP15_inhouse/human_dataset/interface_model_size --modeldir dataset/CASP15_inhouse/human_dataset/models/ --cdpreddir dataset/CASP15_inhouse/human_dataset/cdpred/
python gate/feature/generate_af_scores.py --outdir dataset/CASP15_inhouse/server_dataset/alphafold --indir dataset/CASP15_inhouse/server_dataset/models/ --interface_dir dataset/CASP15_inhouse/server_dataset/interface_model_size/

source /bml/bml_casp15/tools/DPROQ_env/bin/activate base
python gate/feature/generate_dproq_scores.py  --indir /home/jl4mc/gate/dataset/CASP15_inhouse/human_dataset/models/ --outdir /home/jl4mc/gate/dataset/CASP15_inhouse/human_dataset/dproqa --interface_dir /home/jl4mc/gate/dataset/CASP15_inhouse/human_dataset/interface_model_size/

