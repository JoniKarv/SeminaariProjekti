import multiprocessing
import os
import pickle
import urllib.request
import json
import neat
import numpy as np
import Peli


def fetch_values():
   with urllib.request.urlopen('https://elintarvikepeli.herokuapp.com/howmany/100') as response:
      html = response.read()
      JSON = json.loads(html)
      kortti = JSON[0]['salt'], JSON[0]['energyKcal'], JSON[0]['fat'], JSON[0]['protein'], JSON[0]['carbohydrate'], JSON[0]['sugar'], JSON[0]['fiber']
      return JSON

def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0
    i = 0
    #Haetaan peliin tarvittavat kortit
    kortit = fetch_values()
    #Pelataan 50 kierrosta
    while (i < 50):
        #Haetaan pakasta kortti tekoälylle ja vastustajalle
        inputit = kortit[i]['salt'], kortit[i]['energyKcal'], kortit[i]['fat'], kortit[i]['protein'], kortit[i]['carbohydrate'], kortit[i]['sugar'], kortit[i]['fiber']
        kortti = kortit[i + 1]['salt'], kortit[i + 1]['energyKcal'], kortit[i + 1]['fat'], kortit[i + 1]['protein'], kortit[i + 1]['carbohydrate'], kortit[i + 1]['sugar'], kortit[i + 1]['fiber']
        #Annetaan kortti tekoälylle ja vastaanotetaan tekoälyltä syöte
        output = net.activate(inputit)
        #Syöte antaa listan "prioriteetti arvoja", jotka ovat painote arvon suuruudelle. Järjestetään nämä
        #arvot suuruus järjestykseen ja verrataan niiden avulla kaikki ravintoarvot vastustajan korttiin.
        arvot = list(output)
        arvot.sort()
        j = 0
        while (j < len(arvot)):
            vastaus = ""
            loytyi = True
            k = 0
            while (loytyi):
                if (int(arvot[j]) == int(output[k])):
                    #Peli vastaanottaa vastaukseksi numeroja (0-6), joten muunnetaan ravintoaine arvo sen indeksiksi.
                    vastaus = k
                    loytyi = False
                k += 1
            Tulos = Peli.Koulutus(inputit, kortti, vastaus)
            if (Tulos):
                #Jos ravintoarvo voittaa kierroksen niin palkintofunktiolle annetaan pisteitä järjestetyn prioriteetti
                #listan järjestysnumeron (0-6) perusteella kaavalla 2^(järjestysnumero). Eli jos 3. pienin arvo voittaa
                #annetaan 2^2 = 4 pistettä ja jos suurin prioriteetti voittaa niin annetaan 2^6 = 64 pistettä.
                #Tällä tavalla saadaan palautetta kaikista tekoälyn palauttamista arvoista, eikä vain voittavasta arvosta
                fitness += 2 ** j
            j += 1
        i = i + 1
    return fitness


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    with open('winner3', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

if __name__ == '__main__':
    run()
