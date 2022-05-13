import java.util.*;

public class Main {
    public static void main(String[] args) {
        String password = "Deep Learning 2022";

        int populationSize = 16;
        int numOfParents = 2;
        int numOfElites = 2;

        ArrayList<String> population = generatePopulation(populationSize, password.length());
        evolve(population, password, numOfParents, numOfElites);
    }

    public static void evolve(ArrayList<String> population, String password, int numOfParents, int numOfElites) {
        ArrayList<Integer> success = new ArrayList<>();
        int generations = 0;
        long start = System.currentTimeMillis();

        while (true) {
            HashMap<String,Integer> scores = fitness(population, password);
            LinkedHashMap<String,Integer> descendingOrder = new LinkedHashMap<>();

            scores.entrySet()
                    .stream()
                    .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                    .forEachOrdered(x -> descendingOrder.put(x.getKey(), x.getValue()));

            Map.Entry<String,Integer> maxScore = descendingOrder.entrySet().iterator().next();
            success.add(maxScore.getValue());

            if (maxScore.getValue() == password.length()) {
                String found = maxScore.getKey();
                long end = System.currentTimeMillis() - start;

                System.out.println("Generations: " + generations);
                System.out.println("Time taken: " + end + " ms");
                System.out.println("Original password: " + password);
                System.out.println("Discovered password: " + found);
                break;
            }

            ArrayList<String> parents = selectParents(scores, numOfParents);
            ArrayList<String> children = createChildren(parents, population.size(), numOfElites);

            population = mutation(children);
            generations++;
        }
    }
    public static ArrayList<String> generatePopulation(int size, int length) {
        ArrayList<String> population = new ArrayList<>();

        for (int i = 0; i < size; i++) {
            population.add(randomString(length));
        }
        return population;
    }
    public static String randomString(int length) {
        String pool = " ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
        StringBuilder salt = new StringBuilder();
        Random rnd = new Random();

        while (salt.length() < length) {
            int index = (int) (rnd.nextFloat() * pool.length());
            salt.append(pool.charAt(index));
        }

        return salt.toString();
    }
    public static HashMap<String,Integer> fitness(ArrayList<String> population, String password) {
        HashMap<String, Integer> scores = new HashMap<>();

        for (String chromosome: population) {
            int match = 0;

            for (int i = 0; i < chromosome.length(); i++) {
                if (password.charAt(i) == chromosome.charAt(i)) {
                    match += 1;
                }
            }

            scores.put(chromosome, match);
        }

        return scores;
    }
    public static ArrayList<String> selectParents(HashMap<String, Integer> scores, int numOfParents) {
        ArrayList<String> parents = new ArrayList<>();
        LinkedHashMap<String, Integer> descendingOrder = new LinkedHashMap<>();

        scores.entrySet()
                .stream()
                .sorted(Map.Entry.comparingByValue(Comparator.reverseOrder()))
                .forEachOrdered(x -> descendingOrder.put(x.getKey(), x.getValue()));

        int i = 0;
        for (Map.Entry<String, Integer> parent : descendingOrder.entrySet()) {
            if (i < numOfParents) {
                parents.add(parent.getKey());
                i++;
            } else {
                break;
            }
        }

        return parents;
    }
    public static String breed(String parent1, String parent2) {
        String child = "";

        Random rand = new Random();

        int geneA = Math.round(rand.nextFloat() * parent1.length());
        int geneB = Math.round(rand.nextFloat() * parent2.length());

        int start = Math.min(geneA, geneB);
        int end = Math.max(geneA, geneB);

        for (int i = 0; i < parent1.length(); i++) {
            if (i < start || i > end) {
                child = child.concat(Character.toString(parent1.charAt(i)));
            } else {
                child = child.concat(Character.toString(parent2.charAt(i)));
            }
        }

        return child;
    }

    public static ArrayList<String> createChildren(ArrayList<String> parents, int population, int eliteSize) {
        ArrayList<String> children = new ArrayList<>();
        int newChildNum = population - eliteSize;

        Random rand = new Random();

        for (int i = 0; i < eliteSize; i++) {
            children.add(parents.get(i));
        }

        for (int i = 0; i < newChildNum; i++) {

            int rand1 = rand.nextInt(parents.size());
            int rand2 = rand.nextInt(parents.size());

            String parent1 = parents.get(rand1);
            String parent2 = parents.get(rand2);

            String child = breed(parent1, parent2);
            children.add(child);
        }

        return children;
    }

    public static ArrayList<String> mutation(ArrayList<String> children) {
        ArrayList<String> mutated = new ArrayList<>();
        Random rand = new Random();

        for (int i = 0; i < children.size(); i++) {
            if (rand.nextFloat() <= 0.1) {
                int position = rand.nextInt(children.get(0).length());
                char mutation = randomString(1).charAt(0);

                char[] chromosome = children.get(i).toCharArray();
                chromosome[position] = mutation;

                mutated.add(String.valueOf(chromosome));
            } else {
                mutated.add(children.get(i));
            }
        }

        return mutated;
    }
}
