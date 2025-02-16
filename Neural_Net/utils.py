import numpy

class TextProcess:
    def __init__(self):
        char_map_str = """
		' 0
		<SPACE> 1
		a 2
		b 3
		c 4
		d 5
		e 6
		f 7
		g 8
		h 9
		i 10
		j 11
		k 12
		l 13
		m 14
		n 15
		o 16
		p 17
		q 18
		r 19
		s 20
		t 21
		u 22
		v 23
		w 24
		x 25
		y 26
		z 27
		"""
        self.char_map = {}
        self.index_map = {}

        for line in char_map_str.strip().split('\n'):
            ch , index = line.strip().split(" ")
            self.char_map[ch] = int(index)
            self.index_map[int(index)] = ch
        self.index_map[1] = " "

    def text_to_int(self, text):
        sequence = []
        for ch in text:
            if ch == ' ':
                sequence.append(self.char_map['<SPACE>'])
            else:
                sequence.append(self.char_map[ch])
        
        return sequence 

    def int_to_text(self,lables):
        string = []
        for lable in lables:
            string.append(self.index_map[lable])
        
        return "".join(string).replace("<SPACE>", " ")
    


 
        