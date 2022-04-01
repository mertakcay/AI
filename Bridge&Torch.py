#Mert Akçay 
class People:

    def __init__(self,name,speed) -> None:
        self.name = name
        self.speed = speed
        
class AStar:

    def __init__(self,people:list) -> None:
        self.path = []
        self.leftSide = people
        self.rightSide = []
        self.cost = 0

    # final state leftBridge'in boş olması ve rightBridge'in dolu olmasıyla oluşacak burada trick kullanmak gerekirse minimum gidişe sahip olan 2 linin minimumu geri dönerse en az yol ile gitmiş olur.Trick olarakta 2 kişi geçip en hızlı kişinin solda kalmasıyla algoritmada hem işlem hemde kolaylık sağlamış oluruz.
    def run(self):
        
        while(self.leftSide is not None):
            prevCost = 999
            fastNode = None
            slowNode = None
            for i in range(0,len(self.leftSide)):
                for k in range(i+1,len(self.leftSide)):
                    # print(f"leftSide {i} {self.leftSide[i].speed} {self.leftSide[i].name} ")
                    # print(f"leftSide {k} {self.leftSide[k].speed} {self.leftSide[k].name} ")
                    # print("---------------------------------------------------------------")
                    f = self.leftSide[i].speed + self.leftSide[k].speed # rightBridge geçiş maliyeti
                    

                    if(self.leftSide[i].speed> self.leftSide[k].speed): # Min maliyetle tekrar leftBridge geçme ücreti
                        g = self.leftSide[k].speed
                        
                    else:
                        g = self.leftSide[i].speed

                    if(prevCost > (f+g)): # hem left hemde right geçiş ücreti g burada predict değer olarak değerlendiriyorum
                        prevCost = (f+g)
                        if(self.leftSide[i].speed> self.leftSide[k].speed):
                            fastNode = self.leftSide[k] # slowNode sağa geçiceği fastNode solda kalıcağı için burada container içinde tutuyorum 2 sinide tutma amacım sadece print fonksiyonunda göstermek içindir.
                            slowNode = self.leftSide[i]
                            
                        else:
                            fastNode = self.leftSide[i]
                            slowNode = self.leftSide[k]

            if(len(self.leftSide) == 2):
                print(f"Middle Cost: {self.cost} - {slowNode.speed} ")
                self.cost = self.cost + slowNode.speed
                self.path.append(f"Left2Right: {fastNode.name} & {slowNode.name} Cost: {slowNode.speed}")
                print("---------------------")
                print(f"Total Cost: {self.cost}")
                print("---------------------")
                for index,path in enumerate(self.path):
                    print(f"{index} : {path}")
                break
            print(f"Middle Cost: {self.cost} - {slowNode.speed} - {fastNode.speed} ")
            self.cost = self.cost + slowNode.speed + fastNode.speed
            self.path.append(f"Left2Right: {fastNode.name} & {slowNode.name} Cost: {slowNode.speed}")
            self.path.append(f"Right2Left: {fastNode.name} Cost: {fastNode.speed}")

            # print("---------------------------------------")
            # for i in self.leftSide:
            #     print(f"leftSide: {i.name}-{i.speed} ")
            # for i in self.rightSide:
            #     print(f"rightSide: {i.name}-{i.speed} ")
            # print(f"slowNode:: {slowNode.name}-{slowNode.speed}")

            self.leftSide.remove(slowNode)
            self.rightSide.append(slowNode)

            del slowNode
            del fastNode
            prevCost = 999


people = []
people.append(People("Mert",2))
people.append(People("Cem",5)) #
people.append(People("Deniz",7)) #
people.append(People("Begüm",4)) #
people.append(People("Burak",9)) #
people.append(People("Serpil",5)) #
people.append(People("Cem",15)) #
people.append(People("Deniz",17)) #
people.append(People("Begüm",14)) #
people.append(People("Burak",19)) #
people.append(People("Serpil",25)) #
people.append(People("Cem",35)) #
people.append(People("Deniz",27)) #
people.append(People("Begüm",24)) #
people.append(People("Burak",29)) #
people.append(People("Serpil",45)) #

AStar = AStar(people)

AStar.run()

#finish


    




                        
                    


    

        