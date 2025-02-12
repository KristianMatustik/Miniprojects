import numpy as np

# Monte Carlo poker simulations
# 1. func: find overall probabilities of having different combinations
# 2. func: compare strengths of 2 hands, probabilities of winning, losing, splitting of each one
# not very efficient, smart, etc, but was fun, easy and quick to code, also does the job quite good anyway

class Deck52:      
    class Card:
        suites = ['H', 'D', 'S', 'C']
        values = ['2','3','4','5','6','7','8','9','T','J','Q','K','A']
        
        def __init__(self, value, suite):
            self.value=value
            self.suite=suite

        def __str__(self):
            return self.values[self.value]+self.suites[self.suite]
            
        def __lt__(self, other):
            return self.value < other.value

    def __init__(self):
        self.cards=[]

    def __iter__(self):
        self._index = 0
        return self

    def __next__(self):
        if self._index < len(self.cards):
            card = self.cards[self._index]
            self._index += 1
            return card
        else:
            raise StopIteration
        
    def display(self):
        for card in self.cards:
            print(card)

    def init_full_deck(self):
        self.cards.clear()
        for suite in range(len(self.Card.suites)):
            for value in range(len(self.Card.values)):
                self.cards.append(self.Card(value,suite))

    def shuffle(self):
        np.random.shuffle(self.cards)

    def sort(self):
        self.cards=sorted(self.cards)

    def clear(self):
        self.cards.clear()

    def remove_card_values(self, value, suite):
        for card in self.cards:
            if self.Card.values[card.value] == value and self.Card.suites[card.suite] == suite:
                self.cards.remove(card)
                return card
            
    def remove_card(self, rcard):
        for card in self.cards:
            if card==rcard:
                self.cards.remove(card)
                return card

    def remove_top_card(self):
        return self.cards.pop()

    def evaluate_poker(self):
        if len(self.cards)!=7:
            return 0

        suite_counter=[[],[],[],[]] #H,D,S,C
        flush=-1

        last=-1
        counter=-1
        multi=[[],[],[],[]]      #single, pair, tri, quad
        
        straight_counter=1
        straight=-1

        self.sort()

        #PREPROCESS, check combinations
        for i in range(7):                                              
            suite_counter[self.cards[i].suite].append(self.cards[i])    #flush
            if len(suite_counter[self.cards[i].suite])>4:
                flush=self.cards[i].suite

            if (i!=0 and self.cards[i].value != last):                  #pair, twopair, trips, quads, fullhouse
                multi[counter].append(last)
                counter=-1
            counter+=1
            last=self.cards[i].value
            if i==6:
                multi[counter].append(last)

            if i!=0 and self.cards[i].value==self.cards[i-1].value+1:   #straight
                straight_counter+=1
                if straight_counter>4 or (straight_counter==4 and self.cards[i].value==3 and self.cards[6].value==12):
                    straight=self.cards[i].value
            elif self.cards[i].value!=self.cards[i-1].value:
                straight_counter=1


        #EVALUATE combinations, made scoring system (terrible but does the job) to compare any 2 hands
        if straight!=-1 and flush!=-1:                                  #check straightflush
            count=1
            high=-1
            for i in range(1,len(suite_counter[flush])):
                if suite_counter[flush][i].value==suite_counter[flush][i-1].value+1:
                    count+=1
                    if count==4 and suite_counter[flush][i].value==3 and suite_counter[flush][len(suite_counter[flush])-1].value==12:
                        high=3
                    if count>4:
                        high=suite_counter[flush][i].value
                else:
                    count=1
            if high!=-1:
                return 900000000000+100000000*(high+2)


        if len(multi[3])>0:                                             #eval quads
            kicker=0
            if len(multi[0])>0:
                kicker=multi[0][len(multi[0])-1]
            if len(multi[1])>0 and multi[1][len(multi[1])-1]>kicker:
                kicker=multi[1][len(multi[1])-1]
            if len(multi[2])>0 and multi[2][len(multi[2])-1]>kicker:
                kicker=multi[2][len(multi[2])-1]

            return 800000000000+100000000*(multi[3][0]+2)+1000000*kicker

        if (len(multi[2])>0 and len(multi[1])>0) or len(multi[2])>1:    #eval fullhouse
            if len(multi[2])>1:
                pair=multi[2][0]
            else:
                pair=multi[1][len(multi[1])-1]
            return 700000000000+100000000*(multi[2][len(multi[2])-1]+2)+1000000*(pair+2)

        if flush>-1:                                                    #eval flush
            return 600000000000+100000000*(suite_counter[flush][len(suite_counter[flush])-1].value+2)+1000000*(suite_counter[flush][len(suite_counter[flush])-2].value+2) \
                    +10000*(suite_counter[flush][len(suite_counter[flush])-3].value+2)+100*(suite_counter[flush][len(suite_counter[flush])-4].value+2)+1*(suite_counter[flush][len(suite_counter[flush])-5].value+2)

        if straight>-1:                                                 #eval straight
            return 500000000000+100000000*(straight+2)
      
        if len(multi[2])>0:                                             #eval trips
            return 400000000000+100000000*(multi[2][0]+2)+1000000*(multi[0][len(multi[0])-1]+2)+10000*(multi[0][len(multi[0])-2]+2)

        if len(multi[1])>1:                                             #eval twopair
            return 300000000000+100000000*(multi[1][len(multi[1])-1]+2)+1000000*(multi[1][len(multi[1])-2]+2)+10000*(multi[0][len(multi[0])-1]+2)

        if len(multi[1])==1:                                            #eval pair
            return 200000000000+100000000*(multi[1][0]+2)+1000000*(multi[0][len(multi[0])-1]+2)+10000*(multi[0][len(multi[0])-2]+2)+100*(multi[0][len(multi[0])-3]+2)

        else:                                                           #eval high card
            return 100000000000+100000000*(multi[0][len(multi[0])-1]+2)+1000000*(multi[0][len(multi[0])-2]+2) \
                    +10000*(multi[0][len(multi[0])-3]+2)+100*(multi[0][len(multi[0])-4]+2)+1*(multi[0][len(multi[0])-5]+2)       

    
    def add_card(self, card):
        self.cards.append(card)


# Player 1: card1 value = P1_V1, card1 suite = P1_S1, etc. for card 2 player2, num of simulations = n
def pokerHandComparison(P1_V1,P1_S1,P1_V2,P1_S2,P2_V1,P2_S1,P2_V2,P2_S2,n=1000000):
    deck = Deck52()
    hand1 = Deck52()
    hand2 = Deck52()
    community = Deck52()
    win1=0
    win2=0
    deck.init_full_deck()
    hand1.add_card(deck.remove_card_values(P1_V1,P1_S1))
    hand1.add_card(deck.remove_card_values(P1_V2,P1_S2))
    hand2.add_card(deck.remove_card_values(P2_V1,P2_S1))
    hand2.add_card(deck.remove_card_values(P2_V2,P2_S2))
    if len(hand1.cards)!=2 or len(hand2.cards)!=2:
        return

    for i in range(n):
        deck.shuffle()
        for j in range(5):
            card=deck.remove_top_card()
            community.add_card(card)
            hand1.add_card(card)
            hand2.add_card(card)
        eval1=hand1.evaluate_poker()
        eval2=hand2.evaluate_poker()
        if eval1>eval2:
            win1+=1
        elif eval1<eval2:
            win2+=1
        
        for j in range(5):
            card=community.remove_top_card()
            hand1.remove_card(card)
            hand2.remove_card(card)
            deck.add_card(card)
    
    print("Win1:" + str(win1/n*100))
    print("Win2:" + str(win2/n*100))
    print("Split:" + str((n-win1-win2)/n*100))

def pokerHandOdds(n=1000000):
    deck = Deck52()
    hand = Deck52()
    probabilities=[0,0,0,0,0,0,0,0,0]
    for i in range(n):
        hand.clear()
        deck.init_full_deck()
        deck.shuffle()

        for i in range(7):
            hand.add_card(deck.remove_top_card())
        evaluation=hand.evaluate_poker()
        if evaluation<200000000000:
            probabilities[0]+=1
        elif evaluation<300000000000:
            probabilities[1]+=1
        elif evaluation<400000000000:
            probabilities[2]+=1
        elif evaluation<500000000000:
            probabilities[3]+=1
        elif evaluation<600000000000:
            probabilities[4]+=1
        elif evaluation<700000000000:
            probabilities[5]+=1
        elif evaluation<800000000000:
            probabilities[6]+=1
        elif evaluation<900000000000:
            probabilities[7]+=1
        elif evaluation<1000000000000:
            probabilities[8]+=1

    for i in probabilities:
        print(i/n*100)


pokerHandOdds()
pokerHandComparison('A','H','A','S','K','D','K','C',1000000)