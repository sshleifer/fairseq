# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import unittest

import torch

from fairseq import search
from fairseq.sequence_generator import SequenceGenerator
from durbango import pickle_load


import tests.utils as test_utils

FRANCE_ARTICLE =  ' Marseille, France (CNN)The French prosecutor leading an investigation into the crash of Germanwings Flight 9525 insisted Wednesday that he was not aware of any video footage from on board the plane. Marseille prosecutor Brice Robin told CNN that "so far no videos were used in the crash investigation." He added, "A person who has such a video needs to immediately give it to the investigators." Robin\'s comments follow claims by two magazines, German daily Bild and French Paris Match, of a cell phone video showing the harrowing final seconds from on board Germanwings Flight 9525 as it crashed into the French Alps. All 150 on board were killed. Paris Match and Bild reported that the video was recovered from a phone at the wreckage site. The two publications described the supposed video, but did not post it on their websites. The publications said that they watched the video, which was found by a source close to the investigation. "One can hear cries of \'My God\' in several languages," Paris Match reported. "Metallic banging can also be heard more than three times, perhaps of the pilot trying to open the cockpit door with a heavy object.  Towards the end, after a heavy shake, stronger than the others, the screaming intensifies. Then nothing." "It is a very disturbing scene," said Julian Reichelt, editor-in-chief of Bild online. An official with France\'s accident investigation agency, the BEA, said the agency is not aware of any such video. Lt. Col. Jean-Marc Menichini, a French Gendarmerie spokesman in charge of communications on rescue efforts around the Germanwings crash site, told CNN that the reports were "completely wrong" and "unwarranted." Cell phones have been collected at the site, he said, but that they "hadn\'t been exploited yet." Menichini said he believed the cell phones would need to be sent to the Criminal Research Institute in Rosny sous-Bois, near Paris, in order to be analyzed by specialized technicians working hand-in-hand with investigators. But none of the cell phones found so far have been sent to the institute, Menichini said. Asked whether staff involved in the search could have leaked a memory card to the media, Menichini answered with a categorical "no." Reichelt told "Erin Burnett: Outfront" that he had watched the video and stood by the report, saying Bild and Paris Match are "very confident" that the clip is real. He noted that investigators only revealed they\'d recovered cell phones from the crash site after Bild and Paris Match published their reports. "That is something we did not know before. ... Overall we can say many things of the investigation weren\'t revealed by the investigation at the beginning," he said. What was mental state of Germanwings co-pilot? German airline Lufthansa confirmed Tuesday that co-pilot Andreas Lubitz had battled depression years before he took the controls of Germanwings Flight 9525, which he\'s accused of deliberately crashing last week in the French Alps. Lubitz told his Lufthansa flight training school in 2009 that he had a "previous episode of severe depression," the airline said Tuesday. Email correspondence between Lubitz and the school discovered in an internal investigation, Lufthansa said, included medical documents he submitted in connection with resuming his flight training. The announcement indicates that Lufthansa, the parent company of Germanwings, knew of Lubitz\'s battle with depression, allowed him to continue training and ultimately put him in the cockpit. Lufthansa, whose CEO Carsten Spohr previously said Lubitz was 100% fit to fly, described its statement Tuesday as a "swift and seamless clarification" and said it was sharing the information and documents -- including training and medical records -- with public prosecutors. Spohr traveled to the crash site Wednesday, where recovery teams have been working for the past week to recover human remains and plane debris scattered across a steep mountainside. He saw the crisis center set up in Seyne-les-Alpes, laid a wreath in the village of Le Vernet, closer to the crash site, where grieving families have left flowers at a simple stone memorial. Menichini told CNN late Tuesday that no visible human remains were left at the site but recovery teams would keep searching. French President Francois Hollande, speaking Tuesday, said that it should be possible to identify all the victims using DNA analysis by the end of the week, sooner than authorities had previously suggested. In the meantime, the recovery of the victims\' personal belongings will start Wednesday, Menichini said. Among those personal belongings could be more cell phones belonging to the 144 passengers and six crew on board. Check out the latest from our correspondents . The details about Lubitz\'s correspondence with the flight school during his training were among several developments as investigators continued to delve into what caused the crash and Lubitz\'s possible motive for downing the jet. A Lufthansa spokesperson told CNN on Tuesday that Lubitz had a valid medical certificate, had passed all his examinations and "held all the licenses required." Earlier, a spokesman for the prosecutor\'s office in Dusseldorf, Christoph Kumpa, said medical records reveal Lubitz suffered from suicidal tendencies at some point before his aviation career and underwent psychotherapy before he got his pilot\'s license. Kumpa emphasized there\'s no evidence suggesting Lubitz was suicidal or acting aggressively before the crash. Investigators are looking into whether Lubitz feared his medical condition would cause him to lose his pilot\'s license, a European government official briefed on the investigation told CNN on Tuesday. While flying was "a big part of his life," the source said, it\'s only one theory being considered. Another source, a law enforcement official briefed on the investigation, also told CNN that authorities believe the primary motive for Lubitz to bring down the plane was that he feared he would not be allowed to fly because of his medical problems. Lubitz\'s girlfriend told investigators he had seen an eye doctor and a neuropsychologist, both of whom deemed him unfit to work recently and concluded he had psychological issues, the European government official said. But no matter what details emerge about his previous mental health struggles, there\'s more to the story, said Brian Russell, a forensic psychologist. "Psychology can explain why somebody would turn rage inward on themselves about the fact that maybe they weren\'t going to keep doing their job and they\'re upset about that and so they\'re suicidal," he said. "But there is no mental illness that explains why somebody then feels entitled to also take that rage and turn it outward on 149 other people who had nothing to do with the person\'s problems." Germanwings crash compensation: What we know . Who was the captain of Germanwings Flight 9525? CNN\'s Margot Haddad reported from Marseille and Pamela Brown from Dusseldorf, while Laura Smith-Spark wrote from London. CNN\'s Frederik Pleitgen, Pamela Boykoff, Antonia Mortensen, Sandrine Amiel and Anna-Maja Rappard contributed to this report.' # @noqa
SHORTER_ARTICLE = ' (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian territories. The formal accession was marked with a ceremony at The Hague, in the Netherlands, where the court is based. The Palestinians signed the ICC\'s founding Rome Statute in January, when they also accepted its jurisdiction over alleged crimes committed "in the occupied Palestinian territory, including East Jerusalem, since June 13, 2014." Later that month, the ICC opened a preliminary examination into the situation in Palestinian territories, paving the way for possible war crimes investigations against Israelis. As members of the court, Palestinians may be subject to counter-charges as well. Israel and the United States, neither of which is an ICC member, opposed the Palestinians\' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki, speaking at Wednesday\'s ceremony, said it was a move toward greater justice. "As Palestine formally becomes a State Party to the Rome Statute today, the world is also a step closer to ending a long era of impunity and injustice," he said, according to an ICC news release. "Indeed, today brings us closer to our shared goals of justice and peace." Judge Kuniko Ozaki, a vice president of the ICC, said acceding to the treaty was just the first step for the Palestinians. "As the Rome Statute today enters into force for the State of Palestine, Palestine acquires all the rights as well as responsibilities that come with being a State Party to the Statute. These are substantive commitments, which cannot be taken lightly," she said. Rights group Human Rights Watch welcomed the development. "Governments seeking to penalize Palestine for joining the ICC should immediately end their pressure, and countries that support universal acceptance of the court\'s treaty should speak out to welcome its membership," said Balkees Jarrah, international justice counsel for the group. "What\'s objectionable is the attempts to undermine international justice, not Palestine\'s decision to join a treaty to which over 100 countries around the world are members." In January, when the preliminary ICC examination was opened, Israeli Prime Minister Benjamin Netanyahu described it as an outrage, saying the court was overstepping its boundaries. The United States also said it "strongly" disagreed with the court\'s decision. "As we have said repeatedly, we do not believe that Palestine is a state and therefore we do not believe that it is eligible to join the ICC," the State Department said in a statement. It urged the warring sides to resolve their differences through direct negotiations. "We will continue to oppose actions against Israel at the ICC as counterproductive to the cause of peace," it said. But the ICC begs to differ with the definition of a state for its purposes and refers to the territories as "Palestine." While a preliminary examination is not a formal investigation, it allows the court to review evidence and determine whether to investigate suspects on both sides. Prosecutor Fatou Bensouda said her office would "conduct its analysis in full independence and impartiality." The war between Israel and Hamas militants in Gaza last summer left more than 2,000 people dead. The inquiry will include alleged war crimes committed since June. The International Criminal Court was set up in 2002 to prosecute genocide, crimes against humanity and war crimes. CNN\'s Vasco Cotovio, Kareem Khadder and Faith Karimi contributed to this report.'
EXPECTED_SUMMARY_FRANCE = 'French prosecutor says he\'s not aware of any video footage from on board the plane. German daily Bild and French Paris Match claim to have found a cell phone video of the crash. A French Gendarmerie spokesman calls the reports "completely wrong" and "unwarranted" German airline Lufthansa confirms co-pilot Andreas Lubitz had battled depression.'
DESIRED_RESLT_SHORTER =  "The Palestinian Authority becomes the 123rd member of the International Criminal Court. The move gives the court jurisdiction over alleged crimes in Palestinian territories. Israel and the United States opposed the Palestinians' efforts to join the body. But Palestinian Foreign Minister Riad al-Malki said it was a move toward greater justice."
IRAN_ARTICLE = ' (CNN)The United States and its negotiating partners reached a very strong framework agreement with Iran in Lausanne, Switzerland, on Thursday that limits Iran\'s nuclear program in such a way as to effectively block it from building a nuclear weapon. Expect pushback anyway, if the recent past is any harbinger. Just last month, in an attempt to head off such an agreement, House Speaker John Boehner invited Israeli Prime Minister Benjamin Netanyahu to preemptively blast it before Congress, and 47 senators sent a letter to the Iranian leadership warning them away from a deal. The debate that has already begun since the announcement of the new framework will likely result in more heat than light. It will not be helped by the gathering swirl of dubious assumptions and doubtful assertions. Let us address some of these: . The most misleading assertion, despite universal rejection by experts, is that the negotiations\' objective at the outset was the total elimination of any nuclear program in Iran. That is the position of Netanyahu and his acolytes in the U.S. Congress. But that is not and never was the objective. If it had been, there would have been no Iranian team at the negotiating table. Rather, the objective has always been to structure an agreement or series of agreements so that Iran could not covertly develop a nuclear arsenal before the United States and its allies could respond. The new framework has exceeded expectations in achieving that goal. It would reduce Iran\'s low-enriched uranium stockpile, cut by two-thirds its number of installed centrifuges and implement a rigorous inspection regime. Another dubious assumption of opponents is that the Iranian nuclear program is a covert weapons program. Despite sharp accusations by some in the United States and its allies, Iran denies having such a program, and U.S. intelligence contends that Iran has not yet made the decision to build a nuclear weapon. Iran\'s continued cooperation with International Atomic Energy Agency inspections is further evidence on this point, and we\'ll know even more about Iran\'s program in the coming months and years because of the deal. In fact, the inspections provisions that are part of this agreement are designed to protect against any covert action by the Iranians. What\'s more, the rhetoric of some members of Congress has implied that the negotiations have been between only the United States and Iran (i.e., the 47 senators\' letter warning that a deal might be killed by Congress or a future president). This of course is not the case. The talks were between Iran and the five permanent members of the U.N. Security Council (United States, United Kingdom, France, China and Russia) plus Germany, dubbed the P5+1. While the United States has played a leading role in the effort, it negotiated the terms alongside its partners. If the agreement reached by the P5+1 is rejected by Congress, it could result in an unraveling of the sanctions on Iran and threaten NATO cohesion in other areas. Another questionable assertion is that this agreement contains a sunset clause, after which Iran will be free to do as it pleases. Again, this is not the case. Some of the restrictions on Iran\'s nuclear activities, such as uranium enrichment, will be eased or eliminated over time, as long as 15 years. But most importantly, the framework agreement includes Iran\'s ratification of the Additional Protocol, which allows IAEA inspectors expanded access to nuclear sites both declared and nondeclared. This provision will be permanent. It does not sunset. Thus, going forward, if Iran decides to enrich uranium to weapons-grade levels, monitors will be able to detect such a move in a matter of days and alert the U.N. Security Council. Many in Congress have said that the agreement should be a formal treaty requiring the Senate to "advise and consent." But the issue is not suited for a treaty. Treaties impose equivalent obligations on all signatories. For example, the New START treaty limits Russia and the United States to 1,550 deployed strategic warheads. But any agreement with Iran will not be so balanced.  The restrictions and obligations in the final framework agreement will be imposed almost exclusively on Iran. The P5+1 are obligated only to ease and eventually remove most but not all economic sanctions, which were imposed as leverage to gain this final deal. Finally some insist that any agreement must address Iranian missile programs, human rights violations or support for Hamas or Hezbollah.  As important as these issues are, and they must indeed be addressed, they are unrelated to the most important aim of a nuclear deal: preventing a nuclear Iran.  To include them in the negotiations would be a poison pill. This agreement should be judged on its merits and on how it affects the security of our negotiating partners and allies, including Israel. Those judgments should be fact-based, not based on questionable assertions or dubious assumptions.'
SUBWAY_ARTICLE = ' New York (CNN)When Liana Barrientos was 23 years old, she got married in Westchester County, New York. A year later, she got married again in Westchester County, but to a different man and without divorcing her first husband.  Only 18 days after that marriage, she got hitched yet again. Then, Barrientos declared "I do" five more times, sometimes only within two weeks of each other. In 2010, she married once more, this time in the Bronx. In an application for a marriage license, she stated it was her "first and only" marriage. Barrientos, now 39, is facing two criminal counts of "offering a false instrument for filing in the first degree," referring to her false statements on the 2010 marriage license application, according to court documents. Prosecutors said the marriages were part of an immigration scam. On Friday, she pleaded not guilty at State Supreme Court in the Bronx, according to her attorney, Christopher Wright, who declined to comment further. After leaving court, Barrientos was arrested and charged with theft of service and criminal trespass for allegedly sneaking into the New York subway through an emergency exit, said Detective Annette Markowski, a police spokeswoman. In total, Barrientos has been married 10 times, with nine of her marriages occurring between 1999 and 2002.  All occurred either in Westchester County, Long Island, New Jersey or the Bronx. She is believed to still be married to four men, and at one time, she was married to eight men at once, prosecutors say. Prosecutors said the immigration scam involved some of her husbands, who filed for permanent residence status shortly after the marriages.  Any divorces happened only after such filings were approved. It was unclear whether any of the men will be prosecuted. The case was referred to the Bronx District Attorney\'s Office by Immigration and Customs Enforcement and the Department of Homeland Security\'s Investigation Division. Seven of the men are from so-called "red-flagged" countries, including Egypt, Turkey, Georgia, Pakistan and Mali. Her eighth husband, Rashid Rajput, was deported in 2006 to his native Pakistan after an investigation by the Joint Terrorism Task Force. If convicted, Barrientos faces up to four years in prison.  Her next court appearance is scheduled for May 18.'
ARIZONA_ARTICLE = " (CNN)At first police in Marana, Arizona, thought the shoplifted gun Mario Valencia held as he walked through a busy office park was locked and unable to fire. The cable through the lever and trigger couldn't be taken off, an officer was told by an employee of the Walmart where Valencia took the gun and some rounds of ammunition. But just 10 seconds after the worker told police that ... a shot. Valencia had fired into the air, and less than a minute later a police car slammed into him in a move that ended a crime spree and sparked nationwide discussion on the officer's unusual tactic. The 36-year-old Valencia was hospitalized and within a few days transferred to jail where he faces 15 charges, including shoplifting the .30-30 rifle. That February morning, police have said, Valencia committed several crimes in nearby Tucson before stealing a car and driving to the Walmart in Marana. There he went to the sporting goods department, asked to see a rifle, then told an employee he wanted the ammunition. Officer who drove into suspect justified, chief says . The woman told police she gave Valencia the rounds because he told her he would break the case with the bullets inside. He also told her not to do anything stupid. In spite of that she also said she didn't feel threatened, leading police to charge him with shoplifting and not armed robbery. Walmart told CNN's Miguel Marquez that  the store clerk acted appropriately, even using a code to alert security to call police. Valencia took the gun and ammo and fled into a nearby business park where he encountered an officer in a slow-moving patrol car. At one point he pointed the weapon at an officer and at another he pointed it at his head. The officer told him several times to put down the gun, police have said. The officers that were tailing him assumed that he likely couldn't shoot anyone because of the store's lock. Marana police on Thursday said the cable gun lock was still on the rifle when it was recovered. But the wire that goes through the trigger and the lever to reload the gun were loose enough to allow it to still be used, police said. It also should have been wrapped through the lever twice, not once, police said. A Walmart spokesman told CNN that the rifle had been properly locked and might have been affected by the hard blow caused by the police car. Valencia, who is in Pima County Jail, will appear in court again on May 18."


BARCA_ARTICLE = " Lionel Messi, Neymar and Luis Suarez is a star-studded strike force that almost every team in the world would dearly love to have, but when the going gets tough, Barcelona turn to Jeremy Mathieu and Gerard Pique. The French centre-back flung himself through the air to reach Xavi's free kick at the back post, heading home from close range, finally breaching Celta Vigo's stern defence. It was Mathieu who had opened the scoring in the El Clasico, in similar fashion. Pique, meanwhile, just a minute before Mathieu scored, had made an incredible, game-saving tackle on Celta striker Charles, to prevent him from opening the scoring. Jeremy Mathieu (right) heads home in the 73rd minute to hand Barcelona a 1-0 lead against Celta Vigo . Celta Vigo goalkeeper\xa0Alvarez Conde (right) is unable to stop the powerful header from Mathieu . Celta Vigo:\xa0Alvarez Conde, Mallo Novegil, Cabral, Fontas, Castro Otto, Hernandez, Fernandez, Krohn-Dehli (Radoja - 67'), Orellana, Larrivey (Dias de Oliveira 71'), Agudo Duran . Booked:\xa0Krohn-Dehli, Duran . Sent off: Orellana . Subs not used: Mina Lorenzo, Lopez Sanchez, Blanco, Bongonda, Gomez Sola . Barcelona: Bravo, Dani Alves, Pique, Mathieu, Adriano, Rafinha (Xavi 58'), Busquets, Iniesta (Pedro 75'), Messi, Suarez (Rakitic 85'), Neymar . Subs not used:\xa0Ter Stegen, Montoya, Bartra, Sergi . Goals: Mathieu 73' Booked: Suarez . Ref: Inaki Vicandi Garrido . Att: 23,731 . This gritty, barely-deserved triumph helped Barcelona stay four points ahead of Real Madrid at the top of La Liga, after Ancelotti’s side's almost-harrowing 9-1 demolition of Granada . Celta Vigo’s aggressive pressing nearly paid dividends, but their hard-work and effort counted for nothing thanks to Mathieu’s goal. Their frustration manifested itself in a bizarre way, with Fabian Orellana sent off for throwing a lump of turf at Sergio Busquets. For some reason Sergio Alvarez, Celta Vigo's goalkeeper, is kryptonite for Barcelona's attacking trident, as they also failed to beat him in 90 minutes at the Camp Nou earlier this season. After Barcelona's Halloween nightmare against Celta Vigo, the directive from Luis Enrique here was not to leave with egg on their faces on Easter Sunday. Barcelona's own website stated that 'a dose of retribution' was in order, after the 1-0 defeat by Eduardo Berizzo’s side in late autumn. Before the game Spanish media wondered if the international break had happened at just the wrong time for Barcelona and just the right time for Real Madrid – something reflected in Madrid’s big win at lunchtime. The Frenchman wheels away in celebration of his second half header at the Balaidos Stadium . Gerard Pique (right) and Sergio Busquets join the French defender in celebration of his goal . With Los Blancos in disarray, a couple of weeks spent healing wounds away from the Santiago Bernabeu seemed an agreeable plan, while Barcelona's momentum was disrupted. Jordi Alba was injured playing for Spain, so Adriano Correia started in his stead. Messi, meanwhile, was dragged round the United States with Argentina, sidelined because of an inflamed foot which left him unable to put his boots on. He managed to do that here, but his presence was barely noticed on the pitch. With rival Cristiano Ronaldo scoring an incredible five goals earlier, Messi was left frustrated in his quest for the golden boot, with the Madrid man four strikes ahead of him on 36. Luis Enrique hasn't shown himself to be the sentimental type, but the former Celta Vigo manager gave Rafinha Alcantara a rare start against the team which he played for, both before moving to Barcelona at 13, and last season on loan. However, the midfielder failed to impress and made way for the legendary Xavi early in the second half. Mathieu (left) is congratulated by team-mates Lionel Messi and Neymar (right) after his goal . Javier Mascherano was suspended for accumulating too many yellow cards, as is his wont, but luckily for Barcelona Busquets was fit enough to start for the first time in over a month. Alvarez turned in the display of his life at the Camp Nou and Barcelona's attacking trident must have worried it was going to happen again, when he made a brilliant diving stop to deny Lionel Messi. Profligacy was an issue in that 1-0 defeat too, which we were reminded of when Neymar slashed the rebound high and wide. Los Celestes had made the early running themselves and Chilean goalkeeper Claudio Bravo, who boasts the best defensive record in La Liga, was down quickly to deny Joaquin Larrivey at his near post. It was Larrivey that put the first goal of the season past Bravo at the Camp Nou and he was determined not to let the Celta forward in again. With Messi quiet, it was down to the other two prongs of Barcelona's trident to try and make inroads. Neither were successful. Fabian\xa0Orellana was shown a red card for throwing a chunk of grass in the direction of Sergio Busquets . The lump of turf strikes Busquets in the neck after tempers reached boiling point . Neymar antagonised the Celta Vigo fans as he drew several fouls, but the Barcelona ones too with his incredible wastefulness. To watch him in recent weeks is to see a different player to the one that dismantled La Liga defences earlier in the season. His first touch was frequently poor, his passes off-target. Suarez, meanwhile, went down in the box looking for a penalty under an arm from Gustavo Cabral, but far too little contact to actually tumble. He was then booked on the stroke of half-time for a dive, although this time he has been charged into. The one opportunity Suarez got he hit hard and true, stinging the palms of Alvarez. But Celta Vigo were on top and Nolito was getting the better of Dani Alves time and time again. Maybe the Brazilian's mind is elsewhere, with his contract expiring in the summer, but physically he seems less capable too. Orellana (right) leaves the field after referee\xa0Inaki Vicandi Garrido brandishes a red card . The Barcelona players surround referee following a controversial decision . One horrendous defensive mix-up left Nolito charging for the ball with Bravo. The keeper got his foot to it before colliding with the Celta Vigo man, but the fans screamed for a penalty, unsuccessfully. The offside flag being raised against Orellana led to a Celta goal being ruled out, but the best chance fell to Larrivey before the interval. Orellana cut the ball neatly across the box but the striker, unmarked, nine yards out, lacked composure, hacking at it and sending it into the Vigo night sky. Barcelona continued to struggle, with Neymar’s goal harshly ruled out for offside and a shanked Alves shot the only efforts to their credit. Messi and Neymar react after another missed opportunity during a frustrating evening for the duo . Then came Pique’s tackle on Charles, with the defender making a super-human effort to get back and deny the striker. Celta Vigo were almost instantly punished for it, with Mathieu breaking the deadlock. Barcelona couldn’t kill the game, even after Orellana’s late sending off, with Messi scooping the ball over Alvarez but inches over the crossbar too. It didn’t matter; they held on to earn the three points and maintain the four-point lead over Real Madrid that they started the week with. And one game less for each side to play. It’s not pretty, it’s not ‘the Barca way’, but if Enrique lifts La Liga, nobody will be worrying about that. Neymar (right) fights for the ball against Celtais Andreu Fontas during the La Liga clash ."

DEFAULT_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
import os
class TestBart(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = torch.hub.load('pytorch/fairseq', 'bart.large.cnn').eval().to(DEFAULT_DEVICE)
        source_path = "/home/shleifer/transformers_fork/notebooks/test.source"
        if os.path.exists(source_path):
            cls.lns = [" " + x.rstrip() for x in open(source_path).readlines()][:8]
        #cls.lns = pickle_load('/Users/shleifer/transformers_fork/lns.pkl')
        return cls

    def test_bart_batch(self):

        bart = self.model
        tokens = bart.encode(self.lns).to(DEFAULT_DEVICE)
        bart.reset_logs()
        with torch.no_grad():
            bart.extract_features(tokens)
        log_df = bart.combine_logs()
        log_df.to_csv('fairseq_batch_logs.csv')

    def test_bart_gen_batch(self):

        bart = self.model
        bart.reset_logs()
        bart.sample(self.lns, beam=4, lenpen=2.0, max_len_b=140, min_len=55, no_repeat_ngram_size=3)
        log_df = bart.combine_logs()
        log_df.to_csv('fairseq_batch_logs.csv')

    def test_bart_extract_features(self):
        bart = self.model
        text = ' (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian'
        tokens = bart.encode(text).unsqueeze(0).to(DEFAULT_DEVICE)
        bart.reset_logs()
        with self.assertRaises(Exception):
            bart.combine_logs()
        feat = bart.extract_features(tokens)
        bart.log_mem('done')
        log_df = bart.combine_logs()
        log_df.to_csv('fairseq_extract_logs.csv')

    def test_bart_generation(self):
        bart = self.model
        bart.reset_logs()
        with self.assertRaises(Exception):
            bart.combine_logs()
        text = ' (CNN)The Palestinian Authority officially became the 123rd member of the International Criminal Court on Wednesday, a step that gives the court jurisdiction over alleged crimes in Palestinian'
        #tokens = bart.encode(text).unsqueeze(0).to(DEFAULT_DEVICE)
        #gen_default = bart.generate(tokens, max_len_b=20, beam=4)
        
        #gen_text = [bart.decode(g['tokens']) for g in gen_default]
        gen_text = bart.sample([text], beam=4, lenpen=2.0, max_len_b=5, min_len=4,
                                       no_repeat_ngram_size=3)
        bart.log_mem('done')
        print(gen_text)
        log_df = bart.combine_logs()
        log_df.to_csv('fairseq_generate_logs.csv')



    def run_example(self, example):
        bart = self.model
        hypotheses_batch = bart.sample(example, beam=4,
                                       lenpen=2.0, max_len_b=140,
                                       min_len=55, no_repeat_ngram_size=3)

        print(hypotheses_batch)



class TestSequenceGeneratorBase(unittest.TestCase):

    def assertHypoTokens(self, hypo, tokens):
        self.assertTensorEqual(hypo['tokens'], torch.LongTensor(tokens))

    def assertHypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.):
        pos_scores = torch.FloatTensor(pos_probs).log()
        self.assertAlmostEqual(hypo['positional_scores'], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo['tokens'].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel()**lenpen
        self.assertLess(abs(score - hypo['score']), 1e-6)

    def assertAlmostEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertLess((t1 - t2).abs().max(), 1e-4)

    def assertTensorEqual(self, t1, t2):
        self.assertEqual(t1.size(), t2.size(), "size mismatch")
        self.assertEqual(t1.ne(t2).long().sum(), 0)


class TestSequenceGenerator(TestSequenceGeneratorBase):

    def setUp(self):
        self.tgt_dict, self.w1, self.w2, src_tokens, src_lengths, self.model = (
            test_utils.sequence_generator_setup()
        )
        self.sample = {
            'net_input': {
                'src_tokens': src_tokens, 'src_lengths': src_lengths,
            },
        }

    def test_with_normalization(self):
        generator = SequenceGenerator(self.tgt_dict, beam_size=2)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6])

    def test_without_normalization(self):
        # Sentence 1: unchanged from the normalized case
        # Sentence 2: beams swap order
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, normalize_scores=False)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], normalized=False)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], normalized=False)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], normalized=False)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], normalized=False)

    def test_with_lenpen_favoring_short_hypos(self):
        lenpen = 0.6
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, len_penalty=lenpen)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0], lenpen=lenpen)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.9, 0.9, 1.0], lenpen=lenpen)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6], lenpen=lenpen)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.4, 1.0], lenpen=lenpen)

    def test_with_lenpen_favoring_long_hypos(self):
        lenpen = 5.0
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, len_penalty=lenpen)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w2, w1, w2, eos])
        self.assertHypoScore(hypos[0][0], [0.1, 0.9, 0.9, 1.0], lenpen=lenpen)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 1.0], lenpen=lenpen)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, w1, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.4, 1.0], lenpen=lenpen)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.6], lenpen=lenpen)

    def test_maxlen(self):
        generator = SequenceGenerator(self.tgt_dict, beam_size=2, max_len_b=2)
        hypos = generator.generate([self.model], self.sample)
        eos, w1, w2 = self.tgt_dict.eos(), self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w2, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.1, 0.1, 0.6])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.6])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w2, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.3, 0.9, 0.01])


class TestDiverseBeamSearch(TestSequenceGeneratorBase):

    def setUp(self):
        # construct dummy dictionary
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5

        # construct source data
        self.src_tokens = torch.LongTensor([
            [self.w1, self.w2, self.eos],
            [self.w1, self.w2, self.eos],
        ])
        self.src_lengths = torch.LongTensor([2, 2])

        args = argparse.Namespace()
        unk = 0.
        args.beam_probs = [
            # step 0:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.9, 0.1],  # beam 1
                [0.0, unk, 0.9, 0.1],  # beam 2
                # sentence 2:
                [0.0, unk, 0.7, 0.3],
                [0.0, unk, 0.7, 0.3],
            ]),
            # step 1:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [0.0, unk, 0.6, 0.4],
                [0.0, unk, 0.6, 0.4],
                # sentence 2:
                [0.25, unk, 0.35, 0.4],
                [0.25, unk, 0.35, 0.4],
            ]),
            # step 2:
            torch.FloatTensor([
                # eos      w1   w2
                # sentence 1:
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
                # sentence 2:
                [0.9, unk, 0.1, 0.0],
                [0.9, unk, 0.1, 0.0],
            ]),
        ]

        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_diverse_beam_search(self):
        search_strategy = search.DiverseBeamSearch(self.tgt_dict, num_groups=2, diversity_strength=0.)
        generator = SequenceGenerator(
            self.tgt_dict, beam_size=2, search_strategy=search_strategy,
        )
        sample = {'net_input': {'src_tokens': self.src_tokens, 'src_lengths': self.src_lengths}}
        hypos = generator.generate([self.model], sample)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 0.6, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, w1, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 0.6, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.9])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w2, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.4, 0.9])


class TestDiverseSiblingsSearch(TestDiverseBeamSearch):
    def assertHypoScore(
        self, hypo, pos_probs, sibling_rank, diversity_rate, normalized=True, lenpen=1.0
    ):
        pos_scores = torch.FloatTensor(pos_probs).log()
        pos_scores.sub_(torch.Tensor(sibling_rank) * diversity_rate)
        self.assertAlmostEqual(hypo["positional_scores"], pos_scores)
        self.assertEqual(pos_scores.numel(), hypo["tokens"].numel())
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        self.assertLess(abs(score - hypo["score"]), 1e-6)

    def test_diverse_beam_search(self):
        search_strategy = search.DiverseSiblingsSearch(
            self.tgt_dict, diversity_rate=0.5
        )
        generator = SequenceGenerator(
            self.tgt_dict, beam_size=2, search_strategy=search_strategy
        )
        sample = {
            "net_input": {
                "src_tokens": self.src_tokens,
                "src_lengths": self.src_lengths,
            }
        }
        hypos = generator.generate([self.model], sample)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [0.9, 0.6, 1.0], [0, 1, 1], 0.5)
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, w2, eos])
        self.assertHypoScore(hypos[0][1], [0.9, 0.4, 1.0], [0, 2, 1], 0.5)
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w2, eos])
        self.assertHypoScore(hypos[1][0], [0.7, 0.4, 0.9], [0, 1, 1], 0.5)
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w1, eos])
        self.assertHypoScore(hypos[1][1], [0.7, 0.35, 0.9], [0, 2, 1], 0.5)


class TestTopPSamplingSearch(TestSequenceGeneratorBase):

    def setUp(self):
        # construct dummy dictionary
        d = test_utils.dummy_dictionary(vocab_size=2)
        self.assertEqual(d.pad(), 1)
        self.assertEqual(d.eos(), 2)
        self.assertEqual(d.unk(), 3)
        self.eos = d.eos()
        self.w1 = 4
        self.w2 = 5

        # construct source data
        self.src_tokens = torch.LongTensor([
            [self.w1, self.w2, self.eos],
            [self.w1, self.w2, self.eos],
        ])
        self.src_lengths = torch.LongTensor([2, 2])

        args = argparse.Namespace()
        unk = 0.
        # The minimal probability of top 2 tokens.
        self.min_top2_prob = 0.75
        # The minimal probability of the top 1 token.
        self.min_top1_prob = 0.4

        w1_prob = self.min_top1_prob
        w2_prob = self.min_top2_prob - self.min_top1_prob
        eos_prob = 1 - self.min_top2_prob

        args.beam_probs = [
            # step 0:
            torch.FloatTensor([
                # eos      w1   w2
                [0.0, unk, 1.0, 0.0],
                [0.0, unk, 1.0, 0.0],
                [0.0, unk, 1.0, 0.0],
                [0.0, unk, 1.0, 0.0],
            ]),
            # step 1:
            torch.FloatTensor([
                # eos           w1       w2
                [eos_prob, unk, w1_prob, w2_prob],
                [eos_prob, unk, w1_prob, w2_prob],
                [eos_prob, unk, w1_prob, w2_prob],
                [eos_prob, unk, w1_prob, w2_prob],
            ]),
            # step 2:
            torch.FloatTensor([
                # eos      w1   w2
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
                [1.0, unk, 0.0, 0.0],
            ]),
        ]

        task = test_utils.TestTranslationTask.setup_task(args, d, d)
        self.model = task.build_model(args)
        self.tgt_dict = task.target_dictionary

    def test_topp_sampling_search_low_prob(self):
        # Given a prob low enough to top-P sampling, we expect only the top
        # 1 token to be sampled, which always results in the same output.
        low_sampling_topp = self.min_top1_prob/2.0
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=low_sampling_topp)
        generator = SequenceGenerator(
            self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {
            'net_input': {
                'src_tokens': self.src_tokens,
                'src_lengths': self.src_lengths
            }
        }
        hypos = generator.generate([self.model], sample)
        eos, w1 = self.eos, self.w1
        # sentence 1, beam 1
        self.assertHypoTokens(hypos[0][0], [w1, w1, eos])
        self.assertHypoScore(hypos[0][0], [1.0, 0.4, 1.0])
        # sentence 1, beam 2
        self.assertHypoTokens(hypos[0][1], [w1, w1, eos])
        self.assertHypoScore(hypos[0][1], [1.0, 0.4, 1.0])
        # sentence 2, beam 1
        self.assertHypoTokens(hypos[1][0], [w1, w1, eos])
        self.assertHypoScore(hypos[1][0], [1.0, 0.4, 1.0])
        # sentence 2, beam 2
        self.assertHypoTokens(hypos[1][1], [w1, w1, eos])
        self.assertHypoScore(hypos[1][1], [1.0, 0.4, 1.0])

    def test_topp_sampling_search_high_prob(self):
        # Given a prob high enough to top-P sampling, any of the top 2
        # tokens could be sampled. This can cause different outputs.
        high_sampling_topp = (self.min_top1_prob+self.min_top2_prob)/2.0
        search_strategy = search.Sampling(self.tgt_dict, sampling_topp=high_sampling_topp)
        generator = SequenceGenerator(
            self.tgt_dict, beam_size=2, search_strategy=search_strategy)
        sample = {
            'net_input': {
                'src_tokens': self.src_tokens,
                'src_lengths': self.src_lengths
            }
        }
        hypos = generator.generate([self.model], sample)
        eos, w1, w2 = self.eos, self.w1, self.w2
        # sentence 1, beam 1
        self.assertTrue(self.hypoTokens(hypos[0][0], [w1, w1, eos]) or
                        self.hypoTokens(hypos[0][0], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[0][0], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[0][0], [1.0, 0.35, 1.0]))

        # sentence 1, beam 2
        self.assertTrue(self.hypoTokens(hypos[0][1], [w1, w1, eos]) or
                        self.hypoTokens(hypos[0][1], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[0][1], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[0][1], [1.0, 0.35, 1.0]))

        # sentence 2, beam 1
        self.assertTrue(self.hypoTokens(hypos[1][0], [w1, w1, eos]) or
                        self.hypoTokens(hypos[1][0], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[1][0], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[1][0], [1.0, 0.35, 1.0]))

        # sentence 2, beam 2
        self.assertTrue(self.hypoTokens(hypos[1][1], [w1, w1, eos]) or
                        self.hypoTokens(hypos[1][1], [w1, w2, eos]))
        self.assertTrue(self.hypoScore(hypos[1][1], [1.0, 0.4, 1.0]) or
                        self.hypoScore(hypos[1][1], [1.0, 0.35, 1.0]))

    def hypoTokens(self, hypo, tokens):
        return self.tensorEqual(hypo['tokens'], torch.LongTensor(tokens))

    def hypoScore(self, hypo, pos_probs, normalized=True, lenpen=1.):
        pos_scores = torch.FloatTensor(pos_probs).log()
        if not self.almostEqual(hypo['positional_scores'], pos_scores):
            return False
        if pos_scores.numel() != hypo['tokens'].numel():
            return False
        score = pos_scores.sum()
        if normalized:
            score /= pos_scores.numel() ** lenpen
        return abs(score - hypo['score']) < 1e-6

    def almostEqual(self, t1, t2):
        return t1.size() == t2.size() and (t1 - t2).abs().max() < 1e-4

    def tensorEqual(self, t1, t2):
        return t1.size() == t2.size() and t1.ne(t2).long().sum() == 0


if __name__ == '__main__':
    unittest.main()
