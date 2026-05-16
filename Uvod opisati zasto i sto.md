# Završni rad – struktura i sadržaj (Sponge Attacks on Quantized LLMs)

> Ovo je radni kostur (outline) s uputama što napisati u svako poglavlje.
> Kad kreneš pisati finalni tekst, ove upute možeš postupno brisati ili pretvarati u odlomke.

## 1. Uvod — zašto i što radim

### 1.1 Motivacija (problem u praksi)
- Opiši kako se LLM sustavi koriste kao servis (API) s ograničenjima: cijena/kvote po tokenu, rate-limit, ograničenja trajanja i konteksta.
- Objasni da postoji klasa promptova koji namjerno povećavaju računski trošak (latencija, energija, memorija/KV-cache) → praktično DoS na inferenciju.

### 1.2 Što su “sponge” napadi (intuicija)
- Definiraj “spužvaste” (sponge) ulaze: ulazi koji “upijaju” compute/time/energy bez potrebe za privilegiranim pristupom.
- Naglasi da su napadi tipično black-box (ne trebaju gradijente) i mogu raditi samo kroz prompt.

### 1.3 Zašto je kvantizacija bitna
- Kvantizacija (npr. 4-bit) se uvodi radi manjeg memorijskog otiska i bržeg/jeftinijeg izvođenja.
- Međutim, nije očito kako kvantizacija mijenja ranjivost na sponge/DoS: može smanjiti trošak, ali može promijeniti performanse prefill/decode faze, ponašanje KV-cache-a, kernel podršku itd.

### 1.4 Cilj rada (mjerljivo, 1–2 rečenice)
- Primjer formulacije:
	- “Izraditi i evaluirati istraživački alat koji uspoređuje utjecaj sponge/DoS promptova na metrike troška (vrijeme, energija, tokeni) između full-precision i kvantiziranih režima.”
- Po potrebi dodaj: koji modeli, koji kvantizacijski modovi, koje napade obuhvaćaš.

### 1.5 Istraživačka pitanja / hipoteze
- **RQ1:** Povećavaju li se vrijeme/energija inferencije pod napadom u svim kvantizacijskim modovima ili postoje modovi otporniji?
- **RQ2:** Koji tip napada (evolucijski, context exhaustion, AutoDoS) najviše “podiže” trošak i u kojoj fazi (prefill vs decode)?
- **RQ3:** Koliko rezultati ovise o modelu i kontekstu (duljina ulaza, prisiljeni decode)?

### 1.6 Doprinosi
- (a) Implementiran set napada nad lokalnim LLM-ovima.
- (b) Sučelje za pokretanje i usporedbu napada.
- (c) Definiran protokol mjerenja performansi i energije.
- (d) Empirijski rezultati usporedbe fp vs kvantiziranih režima.

### 1.7 Struktura rada
- U 4–6 rečenica: “U poglavlju 2… u poglavlju 3…”.

## 2. Preliminarno (pozadina i pojmovi)

> Cilj: čitatelj razumije izvore troška prije metodologije.

### 2.1 LLM inferencija: prefill i decode
- Prefill: obrada cijelog prompta i izgradnja KV-cache-a.
- Decode: iterativno generiranje tokena (svaki token = novi forward + rad s KV-cache-om).
- Zašto je relevantno za DoS: napadi mogu ciljati prefill (kontekst) ili decode (izlaz).

### 2.2 Kontekstni prozor i KV-cache
- Objasni što znači context window (maks. duljina sekvence).
- KV-cache raste približno linearno s brojem tokena; to povećava memoriju i vrijeme.
- Poveži s “context exhaustion” napadom (ulazi “blizu limita”).
- (Dodatno) Razlikuj nominalni maksimum (MCW) od praktično upotrebljivog konteksta: Paulsen (2025/2026) uvodi pojam *Maximum Effective Context Window* (MECW) i pokazuje da se “učinkovit” kontekst u praksi može značajno razlikovati od deklariranog i ovisi o tipu zadatka.

### 2.3 Metrike troška (što mjeriš i zašto)
- **Trajanje** (s), **tokeni** (ulaz/izlaz), **throughput** (tok/s)
- **Energija** (J):
	- $E = P_{avg} \cdot t$ (W·s = J)
	- Kad power senzor nije dostupan, koristiš vrijeme kao zamjensku metriku (fallback).
- Pomoćne metrike: CPU%, GPU load/temp (primarno kao metadata, ne nužno kao fitness signal).

### 2.4 Kvantizacija (pregled)
- fp16/fp32 vs int8 vs 4-bit (NF4/FP4) vs GPTQ vs GGUF/llama.cpp.
- Tipični dobitci: manja memorija, potencijalno brži inference.
- Trade-off: kvaliteta, kompatibilnost kernela/drivera, razlike u runtimeu.

### 2.5 Prijetnja i etika
- Naglasi da se eksperimenti izvode lokalno i kontrolirano.
- Fokus rada: mjerenje i razumijevanje rizika (defenzivna perspektiva), ne zlouporaba.

## 3. Relevantni radovi (literatura)

> Cilj: pozicionirati tvoj rad i pokazati “gap”.

### 3.1 Sponge napadi na LLM-ove
- Opiši “sponge examples” i black-box optimizaciju (npr. GA) s metrikom energije ili vremena.
- Poveži s time da u implementaciji fitness može biti energija ili vrijeme (fallback).

### 3.2 LLM DoS i prompt-based iscrpljivanje resursa
- Radovi koji ciljaju ekstremno dugačke/skupe odgovore.
- Poveži s motivacijom “compute as attack surface”.

### 3.3 Context exhaustion / KV-cache pressure
- Izvori koji objašnjavaju KV-cache i scaling.
- Uključi kratko objašnjenje kako dulji input utječe na memoriju i vrijeme.
- Uključi i radove o praktičnim limitima konteksta (ne nužno DoS): npr. Paulsen (2025/2026) mjeri degradaciju performansi/uspješnosti s rastom konteksta i definira MECW kao “realni” limit upotrebljivog konteksta.

### 3.4 AutoDoS i srodne metode
- Opiši ideje: attack tree (depth/breadth) i “length trojan” (deklarativno kratak limit, ali zahtjev za iscrpnim odgovorom).

### 3.5 Kvantizacija i sigurnosne implikacije
- Radovi o utjecaju kvantizacije na performanse/robustnost.
- Ako nema direktnog rada “sponge + quant”, jasno reci da tvoj rad daje empirijsku usporedbu.

### 3.6 Sažetak literature i “gap”
- 1 kratka podsekcija: što nedostaje u postojećim radovima i kako tvoj alat/eksperimenti to pokrivaju.

## 4. Metodologija

> Cilj: da se eksperiment može reproducirati.

### 4.1 Pregled sustava / arhitektura alata

Sustav je realiziran kao jednostavna 3-slojna arhitektura: **frontend** (korisničko sučelje), **backend** (API + izvođenje napada) i **baza podataka** (pohrana rezultata za kasniju analizu). Frontend i backend komuniciraju HTTP pozivima, dok se modeli izvršavaju lokalno u backend procesu (CPU/GPU ovisno o dostupnosti i odabranom modu).

#### 4.1.1 Komponente sustava
- **Frontend (web aplikacija)**
	- Omogućuje odabir modela, tipa napada, kvantizacijskog moda i parametara (npr. broj generacija, broj zahtjeva, dubina stabla).
	- Pokreće eksperimente i prikazuje tijek izvođenja (logovi, status, najbolji rezultat) te završne metrike.

- **Backend (API + izvršni sloj)**
	- Izlaže REST API prema frontendu.
	- U pozadini pokreće odabrani napad (npr. evolucijski, context exhaustion, AutoDoS) kao “background task”, tako da UI može ostati responzivan.
	- Napadi su implementirani modularno: svaki tip napada je izdvojen u zasebnu datoteku/modul (npr. `evolutionary_sponge.py`, `context_exhaustion.py`, `autodos_attack.py`), dok backend služi kao orkestrator koji ih poziva kroz jedinstveni API.
	- Učitava model i tokenizer u odabranom režimu (full precision ili kvantizirano) te nakon izvođenja oslobađa memoriju.
	- Provodi mjerenje metrika (vrijeme, tokeni, energija ako je dostupna) i periodički ažurira stanje izvođenja.

- **Baza podataka (pohrana rezultata; opcionalno)**
	- Pohranjuje metapodatke eksperimenta (model, kvantizacija, parametri) i rezultate (metrike + sažetak logova).
	- U ovom projektu pohrana je integrirana preko Supabase REST sučelja (PostgreSQL u pozadini) i aktivira se samo ako su postavljene konfiguracijske varijable okruženja.

#### 4.1.2 Tok podataka (tipičan scenarij izvođenja)
Korisnik u frontendu odabere tip napada, model i kvantizacijski mod, nakon čega frontend prvo dohvaća mogućnosti okruženja (“capabilities”) s backenda (npr. dostupnost GPU-a, podržani kvantizacijski modovi) kako bi onemogućio nedostupne opcije. Zatim frontend šalje zahtjev za pokretanje napada (npr. `POST /api/attack/start`) s odabranim parametrima, a backend pokreće napad u pozadini, učitava model, izvodi napad i tijekom izvođenja ažurira stanje (logovi, napredak, najbolji rezultat). Frontend periodički dohvaća status (npr. `GET /api/attack/status`) i prikazuje tijek izvođenja, a po završetku backend vraća završne metrike i (ako je konfigurirano) sprema rezultat u bazu podataka.

#### 4.1.3 Detekcija sposobnosti (“capabilities”)
- Backend izračunava i vraća sposobnosti okruženja kroz endpoint `GET /api/capabilities`.
- U praksi to uključuje informacije poput: dostupnost GPU-a, naziv/arch GPU-a (ako postoji), dostupnost llama.cpp GGUF backenda i prisutnost potrebnih model datoteka (npr. putanja/direktorij s `.gguf` datotekama).
- Motivacija: smanjuje broj neuspjelih pokretanja (npr. odabran kvantizacijski mod koji nije podržan na toj konfiguraciji).

#### 4.1.4 System Overview (prikupljanje i prikaz podataka)
System Overview prikazuje trenutačno stanje sustava tako da backend prikuplja podatke preko `psutil` i OS sučelja/senzora, a frontend ih periodički dohvaća i vizualizira. Tipično se prikazuju: ukupno i per-core opterećenje CPU-a, zauzeće i ukupna količina RAM-a, stanje diska (slobodno i postotak zauzeća), stanje baterije (postotak, punjenje, procjena preostalog vremena) te termalna očitanja (npr. GPU/SSD/matična ploča) u tabličnom prikazu. Backend te podatke agregira u jednom odgovoru (npr. `GET /api/stats`), a frontend ih osvježava u pravilnim intervalima kako bi korisnik imao “živi” uvid u opterećenje i temperature tijekom izvođenja napada.

#### 4.1.5 Uloga baze podataka u radu
- Baza nije nužna da bi sustav radio, ali povećava **ponovljivost** i olakšava **analizu rezultata** (centralno spremanje parametara + metrika).
- U tekstu rada možeš naglasiti da se radi o istraživačkom alatu: DB služi kao spremište pokusa (experiment log) i omogućuje usporedbu kroz vrijeme i između konfiguracija.

#### 4.1.6 Tehnološki stack (tech stack)
- **Frontend**
	- React uz Vite kao alatni lanac (development server i bundler).
	- JavaScript (SPA pristup) uz brzi HMR tijekom razvoja i optimizirani produkcijski build.
	- Konfiguracija okruženja kroz Vite varijable (npr. prefiks `VITE_`), prema potrebi za URL backend API-ja.

- **Backend**
	- Python 3.10–3.12.
	- FastAPI kao web framework za REST API, uz Uvicorn kao ASGI poslužitelj.
	- PyTorch za izvođenje modela u “torch” runtimeu (CPU/GPU ovisno o dostupnosti).
	- `llama-cpp-python` za izvođenje lokalnih GGUF modela kroz llama.cpp runtime (alternativni backend za inferenciju).
	- `psutil` i OS sučelja (npr. sysfs na Linuxu) za pomoćne metrike i monitoring.
	- `python-dotenv` za učitavanje konfiguracije iz `.env` datoteka.

- **Baza podataka / pohrana**
	- Supabase (PostgreSQL u pozadini) preko REST sučelja za spremanje zapisa o eksperimentima.
	- Aktivira se samo ako su postavljene varijable okruženja (npr. `SUPABASE_URL`, `SUPABASE_API_KEY`).

### 4.2 Napadi (dizajn i implementacija)

Prije pojedinacnih poglavlja, napadi su organizirani kao skup standardiziranih modula koji dijele isti osnovni tijek: odabir modela i kvantizacijskog moda, priprema ulaza prema strategiji napada, izvodenje inferencije uz mjerenje metrika (vrijeme, tokeni, energija ako je dostupna) te povrat rezultata kroz jedinstveni API. Takva struktura omogucuje dosljednu usporedbu napada i lakse dodavanje novih scenarija bez promjena u frontendu.

#### 4.2.1 Evolutionary Sponge

Evolucijski sponge napad oslanja se na ideju “sponge examples” (Shumailov i sur., 2021), gdje se ulazi optimiziraju tako da maksimiziraju trošak inferencije (energija ili vrijeme). U kontekstu LLM-ova, srodna linija rada pokazuje da se genetskim algoritmima mogu inducirati iznimno duga razmisljanja i time povecati compute (Wang i sur., 2026). Ovaj napad zato koristi evolucijski pristup kako bi automatski pronasao “skupe” promptove bez potrebe za gradijentima ili pristupom internim parametrima modela.

U implementaciji je jedinka jedan prompt. Populacija se inicijalizira slucajnim promptovima, a zatim se kroz generacije primjenjuju mutacije i selekcija prema fitnessu. Fitness signal primarno je izmjerena energija ($E = P_{avg} \cdot t$), dok se u slucaju nedostupnih power senzora koristi latencija kao fallback. Svaka jedinka se evaluira tako da se prompt posalje modelu, mjeri se trajanje i tokeni, a SystemMonitor agregira potrebne metrike.

U praksi, ovaj napad je koristan jer je “model-agnostican” (black-box), a rezultati su usporedivi izmedu kvantiziranih i nekvantiziranih modova. Glavno ogranicenje je stohasticnost (ovisnost o seed-u) i relativno velik broj evaluacija potreban da bi se doslo do ekstremnih promptova.

Pseudo-kod (sažetak):

```text
initialize population P with random prompts
for g in 1..G:
	evaluate each prompt p in P -> score(p) using energy or time
	select top-k prompts as elites
	mutate elites to create offspring
	P = elites + offspring
return best prompt and its metrics
```

Kredit: “Sponge Examples: Energy-Latency Attacks on Neural Networks” (Shumailov i sur., 2021) kao temeljna ideja; “Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models” (Wang i sur., 2026) kao LLM-specifični evolucijski pristup.

#### 4.2.2 Context Exhaustion

Context exhaustion napad cilja ogranicenja kontekstnog prozora i KV-cache-a: napadac konstruira ulaz dovoljno dug da izazove maksimalno zauzece memorije i sporiji prefill, a zatim (opcionalno) forsira i decode dio. Iako termin “KV-cache pressure attack” nije standardiziran u literaturi, bliski koncepti se pojavljuju u radovima o prompt-induciranom DoS-u, npr. ReasoningBomb (Liu i sur., 2026), gdje se induciraju patoloski duga razmisljanja koja drasticno povecavaju trosak inferencije. Kao komplementarna pozadina (iako nije DoS rad), Paulsen (2025/2026) pokazuje da se “učinkovit” kontekst (MECW) moze znacajno smanjiti u odnosu na deklarirani maksimum i ovisi o tipu zadatka; u ovom radu to koristimo kao motivaciju da testiramo ponašanje sustava i blizu nominalnih limita konteksta.

U ovoj implementaciji ciljana duljina ulaza se odreduje na temelju kontekstnog limita modela i dostupne memorije (na GPU-u se koristi konzervativna procjena). Napad se moze izvoditi u dva rezima: (a) “prefill_only” gdje se mjeri iskljucivo prefill, i (b) “combined” gdje se nakon prefilla forsira decode na preostali dio konteksta. Ne izvodimo uvijek “combined” jer decode uvodi dodatne konfaundere (stohasticnost uzorkovanja, razlike u duljini/strukturi generiranog teksta, overhead logit post-procesiranja) i cesto dominira ukupnim vremenom, pa se teze izolira cisti efekt pritiska konteksta/KV-cachea u prefilla. “prefill_only” je zato koristan za stabilniju i brzu procjenu skaliranja s duljinom ulaza te za smanjenje rizika od OOM-a pri testiranju blizu limita, dok “combined” sluzi za mjerenje end-to-end najgoreg slucaja kad nas zanima ukupni trosak upita.

Rezultati se biljeze po zahtjevu: trajanje prefilla, trajanje decodea, ukupno trajanje, energija (ako postoji), te izvedene metrike poput latencije po ulaznom tokenu. Ovaj napad je dobar za testiranje “najgoreg slucaja” memorijskog pritiska i stabilnosti runtimea.

Pseudo-kod (sažetak):

```text
target_len = estimate_safe_len(context_limit, free_mem)
for request in 1..N:
	prompt = random_text(target_len)
	try:
		prefill(prompt)
		if mode != prefill_only:
			decode(max_new_tokens)
	except OOM:
		target_len = reduce(target_len) and retry
	record metrics (prefill, decode, energy/time)
```

Kredit: “ReasoningBomb: A Stealthy Denial-of-Service Attack by Inducing Pathologically Long Reasoning in Large Reasoning Models” (Liu i sur., 2026) kao najblizi formalni rad o prompt-induciranom DoS-u kroz ekstremno duga izvajanja.

#### 4.2.3 AutoDoS (tree-based)

AutoDoS napad se temelji na radovima Zhang i sur. (2024) gdje se uvodi “DoS attack tree” i strategija “length trojan”. Ideja je prvo dekomponirati temu u stablo potpitanja (dubina/breadth), zatim prompt omotati tako da deklarativno trazi kratak odgovor, ali u nastavku zahtijeva iscrpne, detaljne odgovore na svako potpitanje. Time se probijaju sigurnosni mehanizmi i generiraju vrlo dugi izlazi.

U implementaciji se za svaki pokus odabire seed tema, generira se stablo pitanja dubine `depth` i sirine `breadth`, a zatim se prompt obogacuje “length trojan” uputom. Rezultat je prompt koji je povrsinski kratko ogranicen, ali semanticki zahtijeva opsezan odgovor. Ogranicenje je kontekstni prozor, pa se ulaz po potrebi skracuje, a generiranje se prilagodava preostalom budzetu tokena.

Parametri `depth` i `breadth` izravno utjecu na broj leaf pitanja i time na duljinu izlaza; povecanje oba parametra tipicno znaci eksponencijalno opterecenje. Ovaj napad je dobar za mjerenje “output heavy” scenarija i robusnosti modela na namjerne napade koji ciljaju dugi odgovor.

Pseudo-kod (sažetak):

```text
topic = sample_topic()
questions = build_attack_tree(topic, depth, breadth)
prompt = apply_length_trojan(questions)
truncate prompt to fit context
generate response with max_new_tokens
record metrics and output length
```

Kredit: “Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings” (Zhang i sur., 2024).

#### 4.2.4 (Opcionalno) Ostali napadi

U nastavku su opisani dodatni napadi koji su implementirani u sustavu. Ako ih ukljucis u evaluaciju, preporucljivo je navesti njihove parametre i zasebne rezultate; u suprotnom ih mozes ostaviti kao kratko dokumentirane scenarije.

##### Token-Busting
Token-busting napadi ciljaju proces tokenizacije: cilj je proizvesti ulaze koji se razbijaju na nenormalno velik broj tokena, sto povecava prefill trosak i moze dovesti do brzeg ispunjavanja kontekstnog prozora. U literaturi se srodni koncepti pojavljuju kao “adversarial tokenization” (Geh i sur., 2025) i “token manipulation” (Schulz i sur., 2025). U ovoj implementaciji prompt se gradi kombinacijom emoji sekvenci, zero-width znakova, kombinatornih dijakritika i razlicitih pisama kako bi se maksimalno “eksplodirala” BPE tokenizacija. Ako broj tokena prekoraci kontekstni limit, ulaz se skracuje.

Pseudo-kod (sažetak):

```text
prompt = generate_bpe_nightmare()
tokens = tokenize(prompt)
if tokens > context_limit: truncate
generate fixed number of output tokens
record tokens, latency, energy/time
```

Kredit: “Adversarial Tokenization” (Geh i sur., 2025) i “TokenBreak: Bypassing Text Classification Models Through Token Manipulation” (Schulz i sur., 2025) kao najblizi formalni radovi.

##### LingoLoop
LingoLoop napadi ciljaju repetitivno, samoodrzivo generiranje gdje model “zapne” u petlji. U literaturi se eksplicitno opisuju tehnike za “state entrapment into endless loops” (Fu i sur., 2026). U ovoj implementaciji prompt izricito zahtijeva ponavljanje iste fraze u nedogled, bez dodatnih objasnjenja, cime se forsira dugi izlaz dokle god to kontekst dopusta. Cilj je mjeriti trosak generiranja dugih, niskoinformacijskih izlaza.

Pseudo-kod (sažetak):

```text
prompt = "repeat phrase LINGO LOOP on every line"
generate up to max_new_tokens
record output length and latency
```

Kredit: “LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops” (Fu i sur., 2026).

##### State Entrapment
State entrapment napadi “zarobljavaju” model u kontradiktorne upute i dugacke interakcije. U ovoj implementaciji simulira se povijest dijaloga s izmjenama sistemskih pravila (npr. “samo JSON”, “samo YAML”, “samo kratka recenica”) i korisnickim zahtjevima koji su u konfliktu s prethodnim pravilima. Model time trosi znacajno vise resursa na uskladivanje i generiranje odgovora kroz vise “turnova”. Ovaj napad je blisko povezan s LingoLoop literaturom (Fu i sur., 2026), gdje se spominje state entrapment kao mehanizam DoS-a.

Pseudo-kod (sažetak):

```text
history = []
for turn in 1..T:
	append system rule and user task
	prompt = build_conversation(history)
	generate response with large max_new_tokens
	append response to history
record total tokens and latency
```

Kredit: “LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops” (Fu i sur., 2026).

### 4.3 Modeli i kvantizacijski modovi (eksperimentalni faktori)

S obzirom na ograničenje od **16 GB VRAM-a**, odabir modela i modova mora biti takav da manji modeli stanu u punoj preciznosti (za referencu), dok se veći oslanjaju isključivo na kvantizaciju. Ovakav postav daje vrlo opširan, slojevit i metodološki čvrst rad.

#### 4.3.1 Odabir modela
U eksperimente uključi tri klase modela:
1. **Mali / Legacy baseline (1.5B - 3B):** `GPT-2 XL (1.5B)` ili `OPT-2.7B`.
   - *Zašto:* Starije arhitekture (bez Grouped-Query Attention) kod kojih KV-cache memorija raste jako brzo. Odlični za jeftino i brzo testiranje "Context Exhaustion" napada i profiliranja.
2. **Moderni standard (7B - 8B):** `Mistral-7B-Instruct-v0.3` (ili `Llama-3-8B-Instruct`).
   - *Zašto:* Optimalna veličina. Ovaj model će u **fp16** preciznosti zauzeti oko 14-15 GB VRAM-a, ostavljajući taman malo prostora za kontekst. Na ovoj veličini možeš napraviti izravnu usporedbu *apsolutno svih* modova (od nekvantiziranog do 2-bitnog) bez prelijevanja u sistemski RAM.
3. **Veliki modeli (14B - 16B):** `Qwen2.5-14B` (ili `Phi-3-Medium 14B`).
   - *Zašto:* Ovi modeli bi u fp16 uzeli oko ~28 GB VRAM-a, što znači da ne stanu na tvoju grafičku. Služe kao studija slučaja gdje je kvantizacija **nužnost**, a ne samo optimizacija. Također omogućuju testiranje ponašanja pri CPU-offloadingu (ako se GGUF ne može cijeli učitati u GPU memoriju pri većim rezolucijama/kontekstima, kako CPU pad performansi utječe na uspjeh DoS napada).

#### 4.3.2 Kvantizacijski modovi
Da bi rad bio sveobuhvatan, testiraj sljedeće tehničke pristupe (npr. na razini 7B modela):

1. **Baseline referenca:**
   - **fp16** (16-bit Float): Potpuna preciznost, služi kao "ground truth" za brzinu, energiju i memoriju (primjenjivo na 1.5B i 7B modele).

2. **GGUF (llama.cpp) format (optimizirano za fleksibilnost):**
   - **Q8_0 (8-bit):** Visoka preciznost s manjim gubitkom. Pokazuje skaliranje kada je memorija samo prepolovljena.
   - **Q4_K_M (4-bit):** "Zlatni standard" kompromisa u industriji. Ovo moraš imati.
   - **IQ2_XXS ili Q2_K (2-bit):** Ekstremna kvantizacija. Zanimljivo za istražit da li napadi potpuno degradiraju (npr. gubi li model "sposobnost" generiranja sponge izlaza jer je previše kvantiziran) te mjerenje uštede energije naspram gubitka kvalitete.

3. **GPU-nativna kvantizacija (HF ekosustav):**
   - **BitsAndBytes NF4 (4-bit):** Standard za učitavanje u PyTorchu/HuggingFaceu (`load_in_4bit=True`). Ponašat će se drugačije memorijski i energetski nego GGUF zbog PyTorch overhead-a.
   - **GPTQ (4-bit, npr. AutoGPTQ):** Optimizirano za visoki throughput izravno na GPU kernelima (striktno GPU izvođenje oslanjajući se na vlažne PyTorch performanse).

*Napomena u metodologiji:* Objasnit ćeš da je Q4_K_M na GGUF-u i NF4 na BitsAndBytes suštinski slična kompresija (4 bita po težini), ali prolaze kroz potpuno drugačije engine i kernele za izvođenje (llama.cpp custom C++/CUDA vs PyTorch), što može dati znatno drukčije energetske profile pri napadu.

### 4.4 Mjerne metrike i instrumentacija
- Kako mjeriš vrijeme, tokene, throughput.
- Kako računaš energiju (power readout → $E=P_{avg}\cdot t$), i što radiš kad power nije dostupan (fallback na vrijeme).
- Koji su izvori očitanja (Linux sysfs/driver izloženost; Windows LHM) i moguće rupe/greške.

### 4.5 Eksperimentalni protokol
- Broj ponavljanja i kontrola slučajnosti (seed) ako postoji.
- Cooldown između runova i warm-up.
- Kontrola varijabli: isti budget tokena, isti `max_new_tokens` / `force_decode_tokens`, isti `context_mode`.
- Kriteriji usporedivosti: npr. energija po input tokenu, latencija po tokenu, “napad vs baseline” omjeri.

### 4.6 Ograničenja metodologije
- Dostupnost power senzora i točnost očitanja.
- Razlike u runtimeovima (torch vs llama.cpp) i kernel podršci (ROCm/CUDA/CPU).
- Utjecaj OS scheduling-a, pozadinskih procesa i termalnog throttlinga.

## 5. Rezultati (uz kratku diskusiju)

> Savjet: strukturiraj po eksperimentima (ili po napadima), ali uvijek navedi postav i metrike.

### 5.1 Postav eksperimenata
- Tablica: hardver, OS, model, kvantizacijski modovi.
- Parametri napada: (gens/pop) ili (num_requests/depth/breadth), te postavke decode-a.

### 5.2 Rezultati po napadu

#### 5.2.1 Evolutionary Sponge
- Graf: best-score kroz generacije.
- Usporedba fp vs quant.
- Prikaži “najskuplji prompt” kao sažetak/uzorak (ne cijeli ako je predugačak).

#### 5.2.2 Context Exhaustion
- Tablica po requestu: `effective_input_tokens`, `prefill_duration`, `decode_duration`, ukupno trajanje, energija.
- Dodatno: `latency_per_input_token_ms` i `energy_per_input_token_mj`.

#### 5.2.3 AutoDoS
- Utjecaj `depth`/`breadth` na trošak.
- Koliko output tokena dobiješ i koliko to košta (vrijeme/energija).

### 5.3 Usporedba kvantizacijskih modova (glavna poruka rada)
- Rangiranje modova po energiji i/ili vremenu pod napadom.
- Omjeri: “napad vs baseline” (koliko puta skuplje).

### 5.4 Analiza i interpretacija
- Gdje se trošak najviše javlja: prefill vs decode.
- Kako kvantizacija mijenja profil (memorija, throughput, stabilnost).
- Kad energija nije dostupna: jasno razdvoji rezultate bazirane na vremenu od onih baziranih na energiji.

### 5.5 Sažetak nalaza
- 5–8 bullet pointova “najvažnije što smo naučili”.

## 6. Zaključci i budući rad

### 6.1 Odgovori na istraživačka pitanja
- Kratko i direktno (1 odlomak po RQ).

### 6.2 Glavni doprinosi
- Što je novi rezultat/artefakt: alat + protokol + empirija.

### 6.3 Ograničenja
- Što bi moglo promijeniti rezultate: senzori, driveri, izbor modela, reproducibilnost.

### 6.4 Pravci budućeg rada
- Proširiti evaluaciju na više model-familyja i hardverskih platformi.
- Dodati i evaluirati mitigacije (rate-limit, detekcija sponge promptova) i izmjeriti trade-off.
- Standardizirati mjerenje energije (vanjski wattmeter ili vendor API) radi boljeg ground-trutha.
- Dublja analiza “zašto” (profiliranje, memorijski tragovi, kernel-level karakteristike).

## 7. Bibliografija

1. Ilia Shumailov, Yiren Zhao, Daniel Bates, Nicolas Papernot, Robert Mullins, Ross Anderson. “Sponge Examples: Energy-Latency Attacks on Neural Networks.” IEEE EuroS&P, 2021. https://arxiv.org/abs/2006.03463
2. Shuqiang Wang, Wei Cao, Jiaqi Weng, Jialing Tao, Licheng Pan, Hui Xue, Zhixuan Chu. “Inducing Overthink: Hierarchical Genetic Algorithm-based DoS Attack on Black-Box Large Language Reasoning Models.” ICML, 2026. https://arxiv.org/abs/2605.13338
3. Xiaogeng Liu, Xinyan Wang, Yechao Zhang, Sanjay Kariyappa, Chong Xiang, Muhao Chen, G. Edward Suh, Chaowei Xiao. “ReasoningBomb: A Stealthy Denial-of-Service Attack by Inducing Pathologically Long Reasoning in Large Reasoning Models.” ACM CCS, 2026. https://arxiv.org/abs/2602.00154
4. Yuanhe Zhang, Zhenhong Zhou, Wei Zhang, Xinyue Wang, Xiaojun Jia, Yang Liu, Sen Su. “Crabs: Consuming Resource via Auto-generation for LLM-DoS Attack under Black-box Settings.” arXiv, 2024. https://arxiv.org/abs/2412.13879
5. Renato Geh, Zilei Shao, Guy Van Den Broeck. “Adversarial Tokenization.” ACL, 2025. https://aclanthology.org/2025.acl-long.1012/
6. Kasimir Schulz, Kenneth Yeung, Kieran Evans. “TokenBreak: Bypassing Text Classification Models Through Token Manipulation.” arXiv, 2025. https://arxiv.org/abs/2506.07948
7. Jiyuan Fu, Kaixun Jiang, Lingyi Hong, Jinglun Li, Haijing Guo, Dingkang Yang, Zhaoyu Chen, Wenqiang Zhang. “LingoLoop Attack: Trapping MLLMs via Linguistic Context and State Entrapment into Endless Loops.” ICLR, 2026. https://arxiv.org/abs/2506.14493
8. Norman Paulsen. “Context Is What You Need: The Maximum Effective Context Window for Real World Limits of LLMs.” arXiv, 2025 (revidirano 2026). https://arxiv.org/abs/2509.21361