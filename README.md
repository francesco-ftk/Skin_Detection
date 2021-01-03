# Skin_Detection
AI exam project

Il progetto consiste nell'addestrare e testare i classificatori Random_Forest e Gaussian_Naive_Bayes nel riconoscimento di pelle nei video. Le implementazioni degli algoritmi di apprendimento sono state prese dalla libreria Python scikit-learn.

Il progetto prevede 3 file Python:
1) Il file "test_and_training_set.py" che include la funzione per salvare frames da video e la funzione che elabora i pixels dei frame per creare i set di training e di test.
2) Il file "table_plot_functions.py" che contiene le funzioni per stampare i risultati sperimentali in grafici e tabelle.
3) Il file "main.py" che usa le funzioni citate sopra per eseguire i test e visualizzarne i risultati.

Prima di eseguire il file "main.py" è necessario creare nella cartella del progetto una directory "Dataset" con al suo interno altre 3 directories "frame_input", "frame_output" e
"Video". Le prime due saranno il luogo dove il file "test_and_training_set.py" scriverà e leggerà i frame mentre l'ultima sarà quella dove saranno memorizzati i video da elaborare.
Un dataset di video contenenti pelle è reperibile alla pagina di Julian Stöttinger, https://feeval.org/Data-sets/Skin_Colors.html. Questo dataset contiene sia video grezzi che 
i rispettivi video già elaborati. Nella directory Video precedentemente creata vanno copiati x video di interesse rinominandoli 1i,1o,2i,2o,...,xi,xo dove "xi" è un video grezzo e
"xo" è il suo corrispettivo video già elaborato. I video vengono letti con l'estensione ".avi".
Una volta eseguito il file "main.py" il programma chiederà di specificare il numero di video grezzi contenuti nella cartella Video, dopodiché eseguirà in autonomia i test stampando i risultati a video.





