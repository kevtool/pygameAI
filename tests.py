import statistics
from game import Game
from algorithm import Algorithm

# check if get_desc_list is working
def desc_list_test():
    model = Algorithm(1)
    desc_list = model.get_desc_list(13, 6)
    assert desc_list == [5, 3, 2, 1, 1, 1]

# check if the next generation has the appropriate weights from its predecessors
def next_gen_weight_test():
    starting_brains = 10
    brains_per_gen = 12

    game = Game()
    model = Algorithm(starting_brains)

    for index, brain_ in enumerate(model.brains):
        print(brain_.layers[0].weights)

        scores, dir_changes = game.run(10, mode='ai', brain=brain_)
        scores, dir_changes = statistics.fmean(scores), statistics.fmean(dir_changes)
        print(index, scores, dir_changes)
        model.record_score(index, scores, dir_changes)

    model.create_new_gen(brains_per_gen)
    model.print_brains_weights()


def main():
    desc_list_test()
    next_gen_weight_test()

if __name__ == "__main__":
    main()