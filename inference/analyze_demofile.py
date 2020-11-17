
import subprocess


dem_path = "/home/marcel/projetos/data/csgo_analyze/experiment/inference_replay/match2.dem"
dest_path = "/home/marcel/projetos/data/csgo_analyze/experiment/inference_replay/"
tick_rate = "64"


process = subprocess.Popen(["/home/marcel/projetos/csgo_parser/print_map_sequence","-mode=file",dem_path
                            ,dest_path
                            ,tick_rate],
                     stdout=subprocess.PIPE,
                     stderr=subprocess.PIPE)
stdout, stderr = process.communicate()
print(stdout,stderr)
