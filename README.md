# Senior Design Project
[Generated Song](https://vocaroo.com/embed/fHoORQk0V9G)
# LSTM Music Generation
`The input to the Recurrent Neural Network is a sequence of four chords. Each value is a chord mapped to a specific integer value ranged 0 to 23.`

`The RNN generates a sequence of 16 notes that correspond to the four chords that were given as input.`

`The use of the RNN was a specific design choice. I chose it because with this problem I'm specifically working with temporal data. This is data that has a sequential ordering as well non-independent instances over time. For this specific problem, every note value is directly dependent on the values of the notes around it. A Recurrent Neural Network is able to capture this information in a way that non-temporal Neural Networks are not be able to.`

`My second design choice was the use of LSTM (Long Short Term Memory) layers inside the RNN. LSTM layers allows the network to be able to handle long-term time dependencies. The first design of this block did not include LSTM layers. After testing the network showed that it lost its context around the 10th note it generated. After which the network generated the exact same value for the rest of the 6 notes. To combat this, the vanilla RNN layers were replaced with LSTM layers. This allowed the Neural Network to keep track of long-term dependencies and keep its context all the way up to the 16th note.`

`The current design of the Neural Network includes 3 LSTM layers with 256 nodes at each layer, which feeds into 3 Dense layers with 512 nodes at each of those layers. There are currently 3 Neural Networks each trained on a different genre of music. The current genres of music are Pop, Blues, and Classical.`

# Pop
![Pop](https://raw.githubusercontent.com/vee-upatising/SDP/master/model1.JPG)
# Blues
![Blues](https://raw.githubusercontent.com/vee-upatising/SDP/master/model2.JPG)
# Classical
![Classical](https://raw.githubusercontent.com/vee-upatising/SDP/master/model3.JPG)



