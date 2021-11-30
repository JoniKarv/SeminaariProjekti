import urllib.request
import json
import os
import random
import pickle
import neat
import random
from tabulate import tabulate

pisteet = 0

def fetch_values():
   with urllib.request.urlopen('https://elintarvikepeli.herokuapp.com/howmany/2') as response:
      html = response.read()
      JSON = json.loads(html)
      kortti = JSON[0]['salt'], JSON[0]['energyKcal'], JSON[0]['fat'], JSON[0]['protein'], JSON[0]['carbohydrate'], JSON[0]['sugar'], JSON[0]['fiber']
      return JSON

def fetch_sata():
   with urllib.request.urlopen('https://elintarvikepeli.herokuapp.com/howmany/101') as response:
      html = response.read()
      JSON = json.loads(html)
      return JSON
   
def sataKierrosta(genome, config, kortti, kortti2):
   net = neat.nn.FeedForwardNetwork.create(genome, config)
   output = net.activate(kortti)
   isoinIndeksi = 0
   suurinArvo = int(output[0])
   if(int(output[1]) > suurinArvo):
      isoinIndeksi = 1
      suurinArvo = int(output[1])
   if(int(output[2]) > suurinArvo):
      isoinIndeksi = 2
      suurinArvo = int(output[2])
   if(int(output[3]) > suurinArvo):
      isoinIndeksi = 3
      suurinArvo = int(output[3])
   if(int(output[4]) > suurinArvo):
      isoinIndeksi = 4
      suurinArvo = int(output[4])
   if(int(output[5]) > suurinArvo):
      isoinIndeksi = 5
      suurinArvo = int(output[5])
   if(int(output[6]) > suurinArvo):
      isoinIndeksi = 6
      suurinArvo = int(output[6])
   valinta = isoinIndeksi
   return Koulutus(kortti, kortti2, valinta)

#Valmiin tekoälyn lataaminen ja pelataan sillä 100 kierrosta
def replay_genome(config_path, genome_path="winner2"):
   config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

   with open(genome_path, "rb") as f:
       genome = pickle.load(f)

   i = 0
   pisteet = 0
   vpisteet = 0
   kortit = fetch_sata()
   while(i < 100):
      kortti = kortit[i]['salt'], kortit[i]['energyKcal'], kortit[i]['fat'], kortit[i]['protein'], kortit[i]['carbohydrate'], kortit[i]['sugar'], kortit[i]['fiber']
      kortti2 = kortit[i + 1]['salt'], kortit[i + 1]['energyKcal'], kortit[i + 1]['fat'], kortit[i + 1]['protein'], kortit[i + 1]['carbohydrate'], kortit[i + 1]['sugar'], kortit[i + 1]['fiber']
      if (sataKierrosta(genome, config, kortti, kortti2)):
         pisteet += 1
      else:
         vpisteet += 1
      i += 1

   config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)


   i = 0
   pisteet2 = 0
   vpisteet2 = 0
   while(i < 100):
      valinta = random.randint(0, 6)
      kortti = kortit[i]['salt'], kortit[i]['energyKcal'], kortit[i]['fat'], kortit[i]['protein'], kortit[i]['carbohydrate'], kortit[i]['sugar'], kortit[i]['fiber']
      kortti2 = kortit[i + 1]['salt'], kortit[i + 1]['energyKcal'], kortit[i + 1]['fat'], kortit[i + 1]['protein'], kortit[i + 1]['carbohydrate'], kortit[i + 1]['sugar'], kortit[i + 1]['fiber']
      if (Koulutus(kortti, kortti2, valinta)):
         pisteet2 += 1
      else:
         vpisteet2 += 1
      i += 1


#Kierroksen tulokseen käytettävä funktio
def Koulutus(kortti, kortti2, valitse):
   if (int(kortti2[int(valitse)]) < int(kortti[int(valitse)])):
#      print("Voitit")
      return True
   else:
#      print("Hävisit")
      return False

def PelaaKierros(genome, config, kortti, kortti2):
   net = neat.nn.FeedForwardNetwork.create(genome, config)
   output = net.activate(kortti)
   isoinIndeksi = 0
   suurinArvo = int(output[0])
   if(int(output[1]) > suurinArvo):
      isoinIndeksi = 1
      suurinArvo = int(output[1])
   if(int(output[2]) > suurinArvo):
      isoinIndeksi = 2
      suurinArvo = int(output[2])
   if(int(output[3]) > suurinArvo):
      isoinIndeksi = 3
      suurinArvo = int(output[3])
   if(int(output[4]) > suurinArvo):
      isoinIndeksi = 4
      suurinArvo = int(output[4])
   if(int(output[5]) > suurinArvo):
      isoinIndeksi = 5
      suurinArvo = int(output[5])
   if(int(output[6]) > suurinArvo):
      isoinIndeksi = 6
      suurinArvo = int(output[6])
   valinta = isoinIndeksi
   print(tabulate([['Suola:', str(kortti[0]), str(output[0])],
      ["Energia (kcal): ", str(kortti[1]), str(output[1])],
      ["Rasva: ", str(kortti[2]), str(output[2])],
      ["Proteiini: ", str(kortti[3]), str(output[3])],
      ["Hiilihydraatti: ", str(kortti[4]), str(output[4])],
      ["Sokeri: ", str(kortti[5]), str(output[5])],
      ["Kuitu: ", str(kortti[6]), str(output[6])]],
      headers=['Ravintoarvo', 'arvo', "Tekoälyn output"]))
#   print(" Suola: ", str(kortti[0]), "\n", "Energia (kcal): ", str(kortti[1]), "\n", "Rasva: ", str(kortti[2]), "\n", "Proteiini: ", str(kortti[3]), "\n", "Hiilihydraatti: ", str(kortti[4]), "\n", "Sokeri: ", str(kortti[5]), "\n", "Kuitu: ", str(kortti[6]))
#   print(output)
   print(valinta)
   return Koulutus(kortti, kortti2, valinta)

      #Valmiin tekoälyn lataaminen ja pelataan sillä 100 kierrosta

def Kierros(config_path, genome_path="winner8"):
   config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction, neat.DefaultSpeciesSet, neat.DefaultStagnation, config_path)

   with open(genome_path, "rb") as f:
       genome = pickle.load(f)

   i = 0
   pisteet = 0
   vpisteet = 0
   kortit = fetch_sata()
#   kortti = [100, 10, 20, 50, 15, 15, 3]
   kortti = kortit[i]['salt'], kortit[i]['energyKcal'], kortit[i]['fat'], kortit[i]['protein'], kortit[i]['carbohydrate'], kortit[i]['sugar'], kortit[i]['fiber']
   kortti2 = kortit[i + 1]['salt'], kortit[i + 1]['energyKcal'], kortit[i + 1]['fat'], kortit[i + 1]['protein'], kortit[i + 1]['carbohydrate'], kortit[i + 1]['sugar'], kortit[i + 1]['fiber']

   if (PelaaKierros(genome, config, kortti, kortti2)):
      pisteet += 1
   else:
      vpisteet += 1
   print(pisteet)

#Kommentoi nämä jos koulutat tekoälyä!!!!
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config.txt')
replay_genome(config_path)
#replay_genome2(config_path)

#Kierros(config_path)