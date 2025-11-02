import utils_gen as utils

generator = utils.RandomLattice(path='./generated')

labels = []
no_samples_to_be_generated = 10000
for i in range(no_samples_to_be_generated):
    lattice, label = generator.generate_lattice(utils.random_wallpaper())
    labels.append({"file_id": i, "label" : label})
    crystal_render = generator.render_lattice(lattice, sample_name=str(i), title=False)
    generator.fourier_trasform(255-crystal_render, sample_name=str(i))
generator.dump_labels(labels)
