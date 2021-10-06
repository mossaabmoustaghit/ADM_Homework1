#--------------------------------------------------------Problem 1 -------------------------------------------------------
# ---------Introduction----------- 
# Say "Hello, World!" With Python
if __name__ == '__main__':
    print ("Hello, World!")
# Python If-Else
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    n = int(raw_input().strip())
if n % 2 == 1 :
    print("Weird")
if n % 2 == 0 & n>= 2 & n<=5:
    print ("Not weird")
if n % 2 == 0 & n>= 6 & n<=20:
    print ("Weird")
if n % 2 == 0 & n>20:
    print("Not weird")
# Arithmetic Operators
if __name__ == '__main__':
    a = int(input())
    b = int(input())
    summ = a+b
    dif = a-b
    pro = a*b
    print(summ)
    print(dif)
    print(pro)
# Python: Division    
if __name__ == '__main__':
  a = int(input())
  b = int(input())
  inte= a//b
  fl=a/b
  print(inte)
  print(fl)
# Loops
if __name__ == '__main__':
    n = int(input())
    i=0;
    while i<n:
        print(i**2)
        i=i+1
        
# Write a function
def is_leap(year):
    leap = False
    
    if year%4==0:
        if year%100==0:
            if year%400==0:
                leap=True
            else:
                leap
        else:
            leap=True
    else:
        leap
                        
    return leap

year = int(input())
print(is_leap(year))

# Print Function
if __name__ == '__main__':
    n = int(input())
    for i in range (1,n+1):
        print(i,end='')
     
# ----------Basic Data Types--------
# List Comprehensions
if __name__ == '__main__':
    x = int(input())
    y = int(input())
    z = int(input())
    n = int(input())
    result=[]
    for X in range(x+1):
        for Y in range(y+1):
            for Z in range(z+1):
                if X+Y+Z != n:
                    perm = [X,Y,Z]
                    result.append(perm)
print(result)
# Find the Runner-Up Score!
if __name__ == '__main__':
    n = int(input())
    arr = map(int, input().split())
    arrlist=list(set(arr)) #without doubles
    listsorted=sorted(arrlist)
    print (listsorted[-2])
# Nested Lists
if __name__ == '__main__':
    listscorename=[]
    for _ in range(int(input())):
        name = input()
        score = float(input())
        listscorename.append([name,score])
    listscorename1=sorted(listscorename,key=lambda x: x[1])
    listscore=sorted(list(set([x[1] for x in listscorename1])))
    seclow=listscore[1]
    names = []
    for y in listscorename:
        if y[1] == seclow:
            names.append(y[0])
    print("\n".join(sorted(names)))
# Finding the percentage
if __name__ == '__main__':
    n = int(input())
    student_marks = {}
    for _ in range(n):
        name, *line = input().split()
        scores = list(map(float, line))
        student_marks[name] = scores
    query_name = input()
    student=list(student_marks[query_name])
    summ=0
    for x in range (len(student)):
        summ=summ+student[x]
    avg=summ/len(student)
    print( '%.2f' %avg)
# Lists
if __name__ == '__main__':
    N = int(input())
    l=[]
    for i in range(0,N):
        inp = input().split();
        if inp[0] == "print":
            print(l)
        elif inp[0] == "insert":
            l.insert(int(inp[1]),int(inp[2]))
        elif inp[0] == "remove":
            l.remove(int(inp[1]))
        elif inp[0] == "pop":
            l.pop();
        elif inp[0] == "append":
            l.append(int(inp[1]))
        elif inp[0] == "sort":
            l.sort();
        else:
            l.reverse();
# Tuples 
if __name__ == '__main__':
    n = int(raw_input())
    integer_list = map(int, raw_input().split())
    print hash(tuple(integer_list))
# -----------Strings-------
# sWAP cASE
def swap_case(s):
    string = ""
    for letter in s:
        if letter.isupper() == True:
            string+=(letter.lower())
        else:
            string+=(letter.upper())
    return string
# String Split and Join
def split_and_join(line):
    # write your code here
    line=line.split(" ")
    line='-'.join(line)
    return line
    
if __name__ == '__main__':
    line = input()
    result = split_and_join(line)
    print(result)
# What's Your Name?
def print_full_name(first, last):
    last=last+'!'
    print('Hello',first,last,'You just delved into python.')

if __name__ == '__main__':
    first_name = input()
    last_name = input()
    print_full_name(first_name, last_name)
# Mutations
def mutate_string(string, position, character):
    return string[:position]+character+string[position+1:]
# Find a string
def count_substring(string, sub_string):
    n=len(string)
    m=len(sub_string)
    counter=0
    for i in range(n-m+1):
        if sub_string==string[i:i+m]:
            counter+=1
    return counter
if __name__ == '__main__':
   string = input().strip()
   sub_string = input().strip()
    
   count = count_substring(string, sub_string)
   print(count)
# String Validators
if __name__ == '__main__':
    s = input()
    boolean=False
    for i in range(len(s)):
        if s[i].isalnum():
            boolean=True
    print (boolean)
    boolean=False
    for i in range(len(s)):
        if s[i].isalpha():
            boolean=True
    print (boolean)
    boolean=False
    for i in range(len(s)):
        if s[i].isdigit():
            boolean=True
    print (boolean)  
    boolean=False   
    for i in range(len(s)):
        if s[i].islower():
            boolean=True
    print (boolean)
    boolean=False
    for i in range(len(s)):
        if s[i].isupper():
            boolean=True
    print (boolean)
# Text Alignement
#Replace all ______ with rjust, ljust or center. 

thickness = int(input()) #This must be an odd number
c = 'H'

#Top Cone
for i in range(thickness):
    print((c*i).rjust(thickness-1)+c+(c*i).ljust(thickness-1))

#Top Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))

#Middle Belt
for i in range((thickness+1)//2):
    print((c*thickness*5).center(thickness*6))    

#Bottom Pillars
for i in range(thickness+1):
    print((c*thickness).center(thickness*2)+(c*thickness).center(thickness*6))    

#Bottom Cone
for i in range(thickness):
    print(((c*(thickness-i-1)).rjust(thickness)+c+(c*(thickness-i-1)).ljust(thickness)).rjust(thickness*6))

# Text Wrap
import textwrap

def wrap(string, max_width):
    n=len(string)
    for i in range(0,n+1,max_width):
        para=string[i:i+max_width]
        if len(para)==max_width:
            print(para)
        else:
            return(para)
    return

if __name__ == '__main__':
    string, max_width = input(), int(input())
    result = wrap(string, max_width)
    print(result)
# Designer Door Mat
N, M = map(int, input().split())
for i in range(1, N, 2):
    print(str('.|.' * i).center(M, '-'))
print('WELCOME'.center(M, '-'))
for i in range(N-2, -1, -2):
    print(str('.|.' * i).center(M, '-'))
# String Formatting
def print_formatted(number):
    # your code goes here
    n = len(bin(number)[2:])
    for i in range(1, number+1):
        dec = str(i)
        octa = oct(i)[2:]
        hexa = hex(i)[2:].upper()
        bina = bin(i)[2:]

        print(dec.rjust(n),octa.rjust(n),hexa.rjust(n),bina.rjust(n))
if __name__ == '__main__':
    n = int(input())
    print_formatted(n)
# Alphabet Rangoli 
def print_rangoli(size):
    # your code goes here
    alpha = "abcdefghijklmnopqrstuvwxyz"
    letters=[]
    it=[]
    for i in range(size):
        letters.append(alpha[i])
        it.append(i)
    it = it[:-1]+it[::-1]
    for i in it:
        lin = letters[-(i+1):]
        row = lin[::-1]+lin[1:]
        print("-".join(row).center(size*4-3, "-"))
if __name__ == '__main__':
    n = int(input())
    print_rangoli(n)
# Capitalize 
def solve(s):
    for x in s.split():
        s = s.replace(x,x.capitalize())
    return s
        
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    s = input()

    result = solve(s)

    fptr.write(result + '\n')

    fptr.close()
# The Minion Game 
def minion_game(string):
    # your code goes here
    count1,count2 = 0,0;
    n= len(string)
    for i in range(n):
        if string[i] in "AEIOU":
            count1 += n-i
        else :
            count2 += n-i
    
    if count1 > count2:
        print("Kevin", count1)
    elif count1 < count2:
        print("Stuart",count2)
    elif count1 == count2:
        print("Draw")
    

if __name__ == '__main__':
    s = input()
    minion_game(s)
# Merge the Tools
def merge_the_tools(string, k):
    n=len(string)
    for i in range(0,n, k):
        t = string[i:i+k]
        record=set()
        for i in t:
            if i not in record:
                print(i,end="")
                record.add(i)
        print()
    

if __name__ == '__main__':
    string, k = input(), int(input())
    merge_the_tools(string, k)
# -----------Sets------------  
# Introduction to Sets
def average(array):
    # your code goes here
    l=set(array)
    avr=sum(l)/len(l)
    return avr
if __name__ == '__main__':
    n = int(input())
    arr = list(map(int, input().split()))
    result = average(arr)
    print(result)
# No Idea!
mn = input().split()
m = int(mn[0])
n=int(mn[1])
arr=[]
arr = list(map(int, input().split()))
length=len(arr)
A = set(map(int, input().split()))
B = set(map(int, input().split()))
score=0
for i in range (length):
    if arr[i] in A:
        score += 1
    if arr[i] in B:
        score -= 1

print(score)
# Symmetric Difference
M = int(input())
mset = set(map(int, input().split()))
N = int(input())
nset = set(map(int, input().split()))
mdefn = mset.difference(nset)
ndefm = nset.difference(mset)
differences = ndefm.union(mdefn)
difflist=sorted(list(differences))
for i in range(len(difflist)):
    print(difflist[i])
# Set .add()
N=int(input())
s=set()
for i in range (N):
    s.add(input())
print (len(s))
# Set .discard(), .remove() & .pop()
n = int(input())
s = set(map(int, input().split()))
N = int(input())
for i in range(N):
    commands = input().split()
    if commands[0]=="discard":
        s.discard(int(commands[1]))
    elif commands[0]=="remove":
        s.remove(int(commands[1]))
    else :
        s.pop()
result=sum(list(s))
print(result)

# Set .union() Operation
n = int(input())
nstudents= set(input().split());
b = int(input())
bstudents = set(input().split());
fstudents = nstudents.union(bstudents)
print(len(fstudents))

# Set .intersection() Operation
n=int(input())
nstudents=set(input().split())
b=int(input())
bstudents=set(input().split())
fstudents=nstudents.intersection(bstudents)
print(len(fstudents))

# Set .difference() Operation
n=int(input())
nstudents=set(input().split())
b=int(input())
bstudents=set(input().split())
fstudents=nstudents.difference(bstudents)
print(len(fstudents))

# Set .symmetric_difference() Operation
n=int(input())
nstudents=set(input().split())
b=int(input())
bstudents=set(input().split())
fstudents=nstudents.symmetric_difference(bstudents)
print(len(fstudents))

# Set Mutations
along=int(input())
a=set(map(int, input().split()))
N=int(input())
for i in range(N):
    commands = input().split()
    if commands[0] == 'intersection_update':
        result = set(map(int, input().split()))
        a.intersection_update(result)
    elif commands[0] == 'symmetric_difference_update':
        result = set(map(int, input().split()))
        a.symmetric_difference_update(result)
    elif commands[0] == 'difference_update':
        result = set(map(int, input().split()))
        a.difference_update(result)
    elif commands[0] == 'update':
        result = set(map(int, input().split()))
        a.update(result)
    
print(sum(a))

# The Captain's Room
k=int(input())
s=map(int, input().split())
s=sorted(s)
n=len(s)

for i in range(n):
    if(i != n-1):
        if(s[i]!=s[i-1] and s[i]!=s[i+1]):
            print(s[i])
            break;
    else:
        print(s[i])

# Check Subset
T=int(input())
for i in range(T):
    nA= input()
    A = set(input().split())
    nB = int(input())
    B = set(input().split())
    print(A.issubset(B))
# Check Strict Superset
A=set(input().split())
n=int(input())
result=True
for i in range(n):
    B = set(input().split())
    if not B.issubset(A):
        result=False
print(result)

# ----------- Collections----------
# Collections.Counter()
from collections import Counter
X=int(input())
sizes=Counter([int(x) for x in input().split()])
N=int(input())
money=0
for i in range (N):
    size,price=map(int,input().split())
    if sizes[size]:
        money += price
        sizes[size] -= 1
print(money)

# DefaultDict Tutorial
from collections import defaultdict
n, m = map(int,input().split())
a = defaultdict(list)
for i in range(1, n + 1):
    a[input()].append(str(i))
for j in range(m):
    print(' '.join(a[input()]) or -1)
    
# Collections.namedtuple()
from collections import namedtuple
N=int(input())
colnames = ','.join(input().split())
table = namedtuple('table',colnames)
count = 0
for i in range(N):
    row = input().split()
    line = table(*row)
    count += int(line.MARKS)

print(count/N)

# Collections.OrderedDict()
N=int(input())
from collections import OrderedDict
ordered_dictionary = OrderedDict()
for i in range(N):
    item = input().split()
    itemPrice = int(item[-1])
    itemName = " ".join(item[:-1])
    if(ordered_dictionary.get(itemName)):
        ordered_dictionary[itemName] += itemPrice
    else:
        ordered_dictionary[itemName] = itemPrice
for i in ordered_dictionary.keys():
    print(i, ordered_dictionary[i])

# Collections.deque()
from collections import deque
N=int(input())
d=deque()
for i in range(N):
    command = input().split()
    if(command[0] == 'append'):
        d.append(command[1])
    elif(command[0] == 'appendleft'):
        d.appendleft(command[1])
    elif(command[0] == 'pop'):
        d.pop()
    elif(command[0] == 'popleft'):
        d.popleft()
print(' '.join(d))

# Word Order
from collections import OrderedDict
n=int(input())
d=OrderedDict()
for i in range(n):
    word = input()
    if word in d:
        d[word] +=1
    else:
        d[word] = 1
print(len(d));
for k,m in d.items():
    print(m,end = " ")

# ------------ Date and Time ---------
# Calendar Module 
import calendar
MM,DD,YY = map(int, input().strip().split())

print(calendar.day_name[calendar.weekday(YY,MM,DD)].upper())

# Time Delta
from datetime import datetime
import math
import os
import random
import re
import sys
format_date="%a %d %b %Y %H:%M:%S %z"
# Complete the time_delta function below.
def time_delta(t1, t2):
    t1=datetime.strptime(t1, format_date)
    t2=datetime.strptime(t2, format_date)
    diff=int(abs((t1-t2).total_seconds()))
    return str(diff)
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    t = int(input())

    for t_itr in range(t):
        t1 = input()

        t2 = input()

        delta = time_delta(t1, t2)

        fptr.write(delta + '\n')

    fptr.close()

# -------------Error and Exception-----------
# Exceptions 
T=int(input())
for i in range (T):
    try:
        a, b = input().split()
        print(int(a)//int(b))
    except ValueError as ve:
        print("Error Code:",ve);
    except ZeroDivisionError as ze:
        print("Error Code:",ze);

# -------------Buit-in-------------
# Zipped! 
N,X=map(int, input().split())
listofmarks=[]
avg=0
for i in range (X):
    listofmarks.append(map(float,input().split()))
for x in zip(*listofmarks):
    avg=sum(x)/len(x)
    print(avg)

# Athlete Sort
import math
import os
import random
import re
import sys



if __name__ == '__main__':
    nm = input().split()

    n = int(nm[0])

    m = int(nm[1])

    arr = []

    for _ in range(n):
        arr.append(list(map(int, input().rstrip().split())))

    k = int(input())
    arr.sort(key = lambda i : i[k])
    for i in arr:
        print(*i,sep=' ')


#  ginortS
word=str(input())
lower=[]
upper=[]
even=[]
odd=[]
for l in word:
    if l.islower():
        lower.append(l)
    elif l.isupper():
        upper.append(l)
    elif l.isdigit():
        if int(l)%2==0:
            even.append(l)
        else:
            odd.append(l)
print(''.join(sorted(lower)+sorted(upper)+sorted(odd)+sorted(even)))

#-------------------Python Functionals--------------
# Map and Lambda Functions
cube = lambda x: x**3 # complete the lambda function 

def fibonacci(n):
    # return a list of fibonacci numbers
    fib=[0, 1]
    for i in range(2, n):
        fib.append(fib[i-1] + fib[i-2])
    return(fib[0:n])
if __name__ == '__main__':
    n = int(input())
    print(list(map(cube, fibonacci(n))))
    
# ----------- Regex and Parsing------------
# Detect Floating Point Number 
import re 
T=int(input())
for i in range(T):
    print(re.search(r'^([-\+])?\d*\.\d+$', input()) is not None)
    
# Re.split()
regex_pattern = r"[.,]+"	# Do not delete 'r'.

import re
print("\n".join(re.split(regex_pattern, input())))

# Group(), Groups() & Groupdict()
import re
regex_pattern = r"([a-zA-Z0-9])\1+"
m = re.search(regex_pattern,input())
if m:
    print(m.group(1))
else:
    print(-1)
    
# Re.findall() & Re.finditer()
import re
con='qwrtypsdfghjklzxcvbnm'
vo='aeiou'
reg_pattern=re.findall(r'(?<=['+con+'])(['+vo+']{2,})(?=['+con+'])', input(),re.IGNORECASE)
if reg_pattern:
    for i in reg_pattern:
        print(i)
else:
    print(-1)
    
# Re.start() & Re.end()
import re
s=str(input())
k=str(input())
reg_pattern=re.compile(k)
r=reg_pattern.search(s)
if not r: 
    print("(-1, -1)")
while r:
    print("({0}, {1})".format(r.start(), r.end() - 1))
    r = reg_pattern.search(s,r.start() + 1)

# Regex Substitution
import re
N=int(input())
for i in range (N):
    print(re.sub(r'(?<= )(&&|\|\|)(?= )', lambda x: 'or' if x.group() == '||' else 'and', input()))
    
# Validating Roman Numerals
regex_pattern = r'M{0,3}(C[MD]|D?C{0,3})(X[CL]|L?X{0,3})(I[VX]|V?I{0,3})$'	# Do not delete 'r'.

import re
print(str(bool(re.match(regex_pattern, input()))))

# Validating phone numbers
import re
N=int(input())
for i in range(N):
    number = input()
    if re.match(r'[789]\d{9}$',number):
        print("YES")
    else:
            print("NO")
    
# Validating and Parsing Email Addresses
import re
n=int(input())
for i in range(n):
    name,email=input().split(' ')
    if re.match(r'^<[A-Za-z](\w|-|\.|_)+@[A-Za-z]+\.[A-Za-z]{1,3}>$', email):
        print(name, email)

# Hex Color Code
import re 
N=int(input())
for i in range(N):
    matches = re.findall(r':?.(#[0-9a-fA-F]{6}|#[0-9a-fA-F]{3})', input())
    if matches:
        print(*matches, sep='\n')

# HTML Parser - Part 1
from html.parser import HTMLParser
N=int(input())
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print ('Start :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])
    def handle_endtag(self, tag):
        print ('End   :', tag)
    def handle_startendtag(self, tag, attrs):
        print ('Empty :', tag)
        for ele in attrs:
            print ('->', ele[0], '>', ele[1])
parser = MyHTMLParser()
for i in range(N):
    parser.feed(input())

# HTML Parser - Part 2
from html.parser import HTMLParser

class MyHTMLParser(HTMLParser):
    def handle_comment(self, data):
        if (len(data.split('\n')) != 1):
            print(">>> Multi-line Comment")
        else:
            print(">>> Single-line Comment")
        print(data.replace("\r", "\n"))
    def handle_data(self, data):
        if data.strip():
            print(">>> Data")
            print(data)
    
html = ""       
for i in range(int(input())):
    html += input().rstrip()
    html += '\n'
    
parser = MyHTMLParser()
parser.feed(html)
parser.close()

# Detect HTML Tags, Attributes and Attribute Values
from html.parser import HTMLParser
N=int(input())
class MyHTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        print(tag)
        for attr, value in attrs:
            print("->", attr, ">", value)
    def handle_startendtag(self, tag, attrs):
        print(tag)
        for attr, value in attrs:
            print("->", attr, ">", value)
html=''
for i in range(N):
    html+=input().rstrip() + '\n'
parser = MyHTMLParser()
parser.feed(html)

# Validating UID 
import re
T=int(input())
for i in range(T):
    Number=input()
    if Number.isalnum():
        if re.search(r'(.*[A-Z]){2,}',Number):
            if re.search(r'(.*[0-9]){3,}',Number):
                if re.search(r'.*(.).*\1+.*',Number):
                    print('Invalid')
                else:
                    print('Valid')
            else:
                print('Invalid')
        else:
            print('Invalid')
    else:
        print('Invalid')


# Validating Credit Card Numbers
import re
N=int(input())
pattern = re.compile(r"^"r"(?!.*(\d)(-?\1){3})"r"[456]"r"\d{3}"r"(?:-?\d{4}){3}"r"$")
for i in range(N):
    Number=input()
    print("Valid" if pattern.search(Number) else "Invalid")
    
# Validating Postal Codes
regex_integer_in_range = r"^[1-9][\d]{5}$"	# Do not delete 'r'.
regex_alternating_repetitive_digit_pair = r"(\d)(?=\d\1)"	# Do not delete 'r'.

# Matrix Script
import math
import os
import random
import re
import sys




first_multiple_input = input().rstrip().split()

n = int(first_multiple_input[0])

m = int(first_multiple_input[1])

matrix = []
string=""
for _ in range(n):
    matrix_item = input()
    matrix.append(matrix_item)
for i in range(m):
    for j in range(n):
        string+=matrix[j][i] 
print(re.sub(r'(?<=\w)([^\w\d]+)(?=\w)', ' ', string))

# --------------XML------------
# XML 1 - Find the Score
mport sys
import xml.etree.ElementTree as etree

def get_attr_number(node):
    # your code goes here
    score=len(node.attrib)
    for i in node:
        score+=get_attr_number(i)
    return score
if __name__ == '__main__':
    sys.stdin.readline()
    xml = sys.stdin.read()
    tree = etree.ElementTree(etree.fromstring(xml))
    root = tree.getroot()
    print(get_attr_number(root))

# XML 2 - Find the Maximum Depth
mport xml.etree.ElementTree as etree
maxdepth = 0
def depth(elem, level):
    global maxdepth
    # your code goes here
    if level==maxdepth:
        maxdepth+=1
    for i in elem:
        depth(i, level+1)
if __name__ == '__main__':
    n = int(input())
    xml = ""
    for i in range(n):
        xml =  xml + input() + "\n"
    tree = etree.ElementTree(etree.fromstring(xml))
    depth(tree.getroot(), -1)
    print(maxdepth)

# ----------- Closures and Decorators---------
# Standardize Mobile Number Using Decorators
def wrapper(f):
    def fun(l):
        # complete the function
        f(['+91 ' + n[-10:-5] + ' ' + n[-5:] for n in l])
    return fun

@wrapper
def sort_phone(l):
    print(*sorted(l), sep='\n')

if __name__ == '__main__':
    l = [input() for _ in range(int(input()))]
    sort_phone(l) 

# Decorators 2 - Name Directory 
import operator

def person_lister(f):
    def inner(people):
        # complete the function
        return map(f,sorted(people, key=lambda x: int(x[2])))
    return inner

@person_lister
def name_format(person):
    return ("Mr. " if person[3] == "M" else "Ms. ") + person[0] + " " + person[1]

if __name__ == '__main__':
    people = [input().split() for i in range(int(input()))]
    print(*name_format(people), sep='\n')

    
# -------- Numpy ---------
# Arrays
import numpy

def arrays(arr):
    # complete this function
    # use numpy.array
    return(numpy.array(arr[::-1],float))

arr = input().strip().split(' ')
result = arrays(arr)
print(result)

# Shape and Reshape
import numpy
l=numpy.array(input().split(),int)
print(l.reshape(3,3))

# Transpose and Flatten
import numpy
N,M=map(int,input().split())
arr=[]
for i in range(N):
    arr.append([int(j) for j in input().strip().split()])
arr=numpy.array(arr)
print(numpy.transpose(arr))
print(arr.flatten())

# Concatenate
import numpy
N,M,P=map(int,input().split())
matrix1=[]
matrix2=[]
for i in range (N):
    matrix1.append([int(j) for j in input().split()])
for i in range(M):
    matrix2.append([int(j) for j in input().split()])
matrix1=numpy.array(matrix1)
matrix2=numpy.array(matrix2)
print(numpy.concatenate((matrix1,matrix2),axis=0))

# Zeros and Ones
import numpy
N=tuple(map(int,input().split()))
print(numpy.zeros(N, int),numpy.ones(N, int),sep='\n')

# Eye and Idendity 
import numpy
N,M=map(int,input().split())
A=str(numpy.eye(N,M, k = 0))
A=A.replace('1',' 1')
A=A.replace('0',' 0')
print(A) 

# Array Mathematics
import numpy
N,M=map(int,input().split())

A=[]
B=[]
for i in range (N):
    A.append(input().split())
for i in range (N):
    B.append( input().split())
A=numpy.array(A,int)
B=numpy.array(B,int)
print(A+B)
print(A-B)
print(A*B)
print(A//B)
print(A%B)
print(A**B)

# Floor, Ceil and Rint
import numpy
A=numpy.array(input().split(' '),float)
numpy.set_printoptions(sign=' ')
print(numpy.floor(A))
print(numpy.ceil(A))
print(numpy.rint(A))

# Sum and Prod
import numpy
N,M=map(int, input().split())
A=[]
for i in range (N):
    A.append(input().split())
A=numpy.array(A,int)
sum = numpy.sum(A,axis=0)
prod=numpy.prod(sum,axis=0)
print(prod)

# Min and Max
import numpy
N,M=map(int,input().split())
A=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A,int)
mini=numpy.min(A,axis=1)
print(numpy.max(mini,axis=0))

# Dot and Cross
import numpy
N=int(input())
A=[]
B=[]
for i in range(N):
    A.append(input().split())
for i in range(N):
    B.append(input().split())  
A=numpy.array(A,int)
B=numpy.array(B,int)

print(numpy.dot(A,B))

# Inner And Outer
import numpy
A=numpy.array(input().split(),int)
B=numpy.array(input().split(),int)
print(numpy.inner(A,B))
print(numpy.outer(A,B))

# Polynomials 
import numpy
P=[float(i) for i in input().split()]
x=float(input())
print (numpy.polyval(P, x))

# Linear Algebra
import numpy
N=int(input())
M=[]
for i in range(N):
    M.append(input().split())
M=numpy.array(M,float)

print (round(numpy.linalg.det(M),2))

# Mean, Var, and Std
import numpy
N,M=map(int,input().split())
A=[]
for i in range(N):
    A.append(input().split())
A=numpy.array(A,int)

print(numpy.mean(A,axis=1))
print(numpy.var(A,axis=0))
print(numpy.std(A))
#it doesn't give the right value for std and I can't figure out why.


#----------------------------------Problem 2-----------------------------------------------
# Birthday Cake Candles
import math
import os
import random
import re
import sys

#
# Complete the 'birthdayCakeCandles' function below.
#
# The function is expected to return an INTEGER.
# The function accepts INTEGER_ARRAY candles as parameter.
#

def birthdayCakeCandles(candles):
    # Write your code here
    n=len(candles)
    counter=0
    mheight=max(candles)
    for i in range (n):
        if candles[i]==mheight:
            counter+=1
    return counter        

if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    candles_count = int(input().strip())

    candles = list(map(int, input().rstrip().split()))

    result = birthdayCakeCandles(candles)

    fptr.write(str(result) + '\n')

    fptr.close()



# Number Line Jumps (kangaroo)

import math
import os
import random
import re
import sys

#
# Complete the 'kangaroo' function below.
#
# The function is expected to return a STRING.
# The function accepts following parameters:
#  1. INTEGER x1
#  2. INTEGER v1
#  3. INTEGER x2
#  4. INTEGER v2
#

def kangaroo(x1, v1, x2, v2):
    # Write your code here
    xdiff=x1-x2
    vdiff=v2-v1
    if (x1==x2) and (v1==v2):
        return ('YES')
    elif x1!=x2 and (v1==v2):
        return ('NO')
    elif (xdiff+vdiff)%vdiff==0 and (xdiff+vdiff)/vdiff>0:
        return ('YES')
    else:
        return ('NO')
    
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')

    first_multiple_input = input().rstrip().split()

    x1 = int(first_multiple_input[0])

    v1 = int(first_multiple_input[1])

    x2 = int(first_multiple_input[2])

    v2 = int(first_multiple_input[3])

    result = kangaroo(x1, v1, x2, v2)

    fptr.write(result + '\n')

    fptr.close()
    
    
    









