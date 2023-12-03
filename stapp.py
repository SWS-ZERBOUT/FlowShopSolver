import numpy as np
import pandas as pd
import streamlit as st
from io import StringIO

st.title("FlowShop Solver")
st.header("Input")
input_type = st.selectbox("Choose an input method", [".txt file", "Fill manually"])
if input_type == ".txt file":
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        string_data = stringio.read()
        temps = np.array([list(map(float, row.split())) for row in  string_data.split('\n')])
        nbr_tasks = len(temps[0])
        Columns = []
        for i in range(nbr_tasks):
            Columns.append("T"+str(i+1))
        nbr_machines = len(temps)
        Indexes = []
        for i in range(nbr_machines):
            Indexes.append("M"+str(i+1))
        tableau = pd.DataFrame(temps,columns=Columns,index=Indexes)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(' ')
        with col2:
            st.dataframe(tableau)
        with col3:
            st.write(' ')


if input_type == "Fill manually":
    nbr_machines = st.number_input("Number of machines:", min_value=1, step=1)
    nbr_tasks = st.number_input("Number of tasks:", min_value=1, step=1)
    st.header("Execution time matrix")
    matrix_str = st.text_area("Fill the matrix values (row by row, columns are separed with spaces):")
    try:
        temps = np.array([list(map(float, row.split())) for row in matrix_str.split('\n')])           
        if temps.shape == (nbr_machines, nbr_tasks):
            st.success("Martix loaded successfully:")
            col1, col2, col3 = st.columns(3)
            Columns = []
            for i in range(nbr_tasks):
                Columns.append("T"+str(i+1))
            Indexes = []
            for i in range(nbr_machines):
                Indexes.append("M"+str(i+1))
            tableau = pd.DataFrame(temps,columns=Columns,index=Indexes)

            with col1:
                st.write(' ')

            with col2:
                st.dataframe(tableau)

            with col3:
                st.write(' ')
            
        else:
            st.error(f"Error : The shape of the matrix does not match the specified number of rows ({nbr_machines}) or columns ({nbr_tasks}).")
    except ValueError:
        st.error("Error : Please enter a valid matrix.")

# foncion LPT indices
def indices_descending(arr):
    indexed_array = [(value, index) for index, value in enumerate(arr)]
    sorted_pairs = sorted(indexed_array, key=lambda x: x[0], reverse=True)
    descending_indices = [index for value, index in sorted_pairs]
    return descending_indices
# fonction Johnson
def Johnson(temps):
    Sequence = []
    X = np.where(temps[1,:]>=temps[0,:]) # l'ensemble X
    rest = np.where(temps[1,:]<temps[0,:]) # T\X
    p_X = temps[0][X] 
    p_rest = temps[1][rest]
    SPT_1 = np.argsort(p_X) # SPT 1 
    for i in range(len(X[0])):
        Sequence.append(X[0][SPT_1[i]])
    LPT_2 = indices_descending(p_rest) # LPT 2 
    for i in range(len(rest[0])):
        Sequence.append(rest[0][LPT_2[i]])
    return Sequence
# Procédure dates
def temps_fdt(Sequence,temps):
    C = [[]]
    C[0].append(temps[0][Sequence[0]])
    for i in range(1,len(Sequence)) : C[0].append(C[0][i-1]+temps[0][Sequence[i]]) #for i in range(1,len(temps[0]))
    for j in range(1,len(temps)) :
        c = [temps[j][Sequence[0]] + C[j-1][0]]
        for i in range(1,len(Sequence)): #for i in range(1,len(temps[0]))
            c.append(temps[j][Sequence[i]]+max(c[i-1],C[j-1][i]))
        C.append(c)
    return C
# Heuristique Gupta
def Gupta(temps):   
    e = []
    minimum = []
    s = []
    col_sum_p = []#
    sum_p = []#
    for j in range(nbr_machines-1):
        col_sum_p.append("Pi"+str(j+1)+"+Pi"+str(j+2))#
    for i in range(len(temps[0])):
        if temps[0][i] < temps[-1][i]: e.append(1)
        else : e.append(-1)
        min_value = temps[0][i]+temps[1][i]
        sum = [temps[0][i]+temps[1][i]]
        for j in range(1,len(temps)-1):
            sum.append(temps[j][i]+temps[j+1][i])#           
            if (temps[j][i]+temps[j+1][i]<min_value): min_value = temps[j][i]+temps[j+1][i]
        sum_p.append(sum)#
        minimum.append(min_value)
        s.append(e[i]/min_value)
    # Affichage du tableau
    table = pd.DataFrame(sum_p,columns=col_sum_p,index=Columns)
    table["min"] = minimum
    table["ei"] = e
    table["si"] = s
    st.dataframe(table)
    Sequence = indices_descending(s)
    return Sequence
# Heuristique NEH
def NEH(temps):
    Seq = []
    p = np.array([])
    for i in range(len(temps[0])):
        value = temps[0][i]
        for j in range(1,len(temps)): value += temps[j][i]
        p = np.append(p,value)
    # affichage duree globale
    p_reshaped = p.reshape(1,nbr_tasks)
    table1 = pd.DataFrame(p_reshaped,columns=Columns,index=['Global Pi'])
    st.dataframe(table1)
    LPT = indices_descending(p)
    # affichage LPT
    SEQ = "T"+str(LPT[0]+1)
    for i in range(1,len(LPT)):
        SEQ = SEQ + " ~ " +"T"+str(LPT[i]+1)
    st.write('LPT :', SEQ)
    Seq.append(LPT[0])
    sequence = 'T'+str(LPT[0]+1)
    for k in range(1,len(LPT)):
        sous_seq = []
        score = []
        st.write('Seq = ',sequence)
        st.write('Partial sequences :')
        for i in range(len(Seq)+1):
            seq = np.insert(Seq, i, LPT[k])
            seq_str = 'T'+str(seq[0]+1)
            for i in range(1,len(seq)):
                seq_str = seq_str + ' ~ ' + 'T'+str(seq[i]+1)            
            sous_seq.append(seq)
            C = temps_fdt(seq,temps)
            score.append(C[-1][-1])       
            st.write(seq_str,'; Cmax = ',str(C[-1][-1]))
        minimum = np.argmin(score)
        st.write('Best k =',str(minimum+1))
        st.write('~~~~~~~~~~~~~~')
        Seq = sous_seq[minimum]
        sequence = 'T'+str(Seq[0]+1)
        for i in range(1,len(Seq)):
            sequence = sequence + ' ~ ' + 'T'+str(Seq[i]+1)   
    return Seq
# Heuristique CDS
def CDS(temps):
    Sequence = []
    Cmax = []
    for k in range(1,len(temps)):
        p = [[],[]]
        for i in range(len(temps[0])):
            value1 = temps[0][i]
            value2 = temps[(len(temps)-1)-k+1][i]
            for j in range(1,k): value1 = value1 + temps[j][i]
            for j in range((len(temps)-1)-k+2,len(temps)):value2 = value2 + temps[j][i]
            p[0].append(value1)
            p[1].append(value2)
        p = np.array(p)
        seq = Johnson(p)
        Sequence.append(seq)
        C = temps_fdt(Sequence[-1],temps)
        Cmax.append(C[-1][-1])
        #affichage table de chaque iteration
        table = pd.DataFrame(p,columns=Columns,index=["p'1","p'2"])
        st.write('Step k =',str(k) ,' :')
        st.dataframe(table)
        SEQ = "T"+str(seq[0]+1)
        for i in range(1,len(seq)):
            SEQ = SEQ + " ~ " +"T"+str(seq[i]+1)
        st.write('Sequence : ', SEQ)
        st.write('Makespan :', Cmax[k-1])
    x = np.argmin(Cmax)
    sequence = Sequence[x]

    return sequence

# affichage de la solution
def affichage(sequence,temps):
    SEQ = "T"+str(sequence[0]+1)
    for i in range(1,len(sequence)):
        SEQ = SEQ + " ~ " +"T"+str(sequence[i]+1)
    st.write("The sequence is :",SEQ)
    C = temps_fdt(sequence,temps)
    st.write("Makespan value :",C[-1][-1])
# Résolution
st.header("Minimize Makespan")
if nbr_machines == 2 :
    col1,col2,col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        solve = st.button("Solve with Johnson")
    with col3:
        st.write(" ")
    if solve == True:
        sequence = Johnson(temps)
        affichage(sequence,temps)

elif nbr_machines >=3:
    methode = st.selectbox("Choose the method", ["CDS", "Gupta" , "NEH"])
    col1,col2,col3 = st.columns(3)
    with col1:
        st.write(" ")
    with col2:
        solve = st.button("Solve")
        if solve == True:
            if methode == "CDS":
                sequence = CDS(temps)
            elif methode ==  "Gupta":
                sequence = Gupta(temps)      
            elif methode == "NEH":
                sequence = NEH(temps)
    with col3:
        st.write(" ")
    if solve == True:
        affichage(sequence,temps)

col1,col2,col3 = st.columns(3)
with col1:
    st.write("made by : S.W. ZERBOUT")
with col2:
    st.write(" ")
with col3:
    st.write("Supervised by : Professor BOUDHAR")